#!/usr/bin/env python3
"""
export_onnx.py - Giai đoạn 3.1: Xuất models/best.pt sang ONNX và chuẩn bị đồ thị cho Hailo DFC.

Đầu ra:
  models/best.onnx            Đồ thị đầy đủ (FP32, opset 11, static 1x3x640x640, đã simplify).
                              Đầu ra (1, 4 + nc, 8400) = box xywh đã decode + xác suất lớp.
                              KHÔNG chứa NMS -> dùng để đánh giá ONNX Runtime trên server.
  models/best_hailo.onnx      Đồ thị đã cắt tại 6 Conv cuối của Detect head - chỉ còn backbone,
                              neck và head thuần tích chập. Phần decode DFL (Reshape/Softmax/
                              Transpose), make_anchors, sigmoid và NMS được chuyển sang
                              nms_postprocess của Hailo hoặc CPU.
  models/best_hailo_meta.json
                              end_node_names, shape các đầu ra, stride, reg_max, chuẩn hóa đầu vào
                              -> dùng cho optimization/compile_npu.py và edge_deployment/.

Về `nms=False`:
  Trong Ultralytics 8.4, `nms` là tham số 3 trạng thái: None = NMS ngoài (mặc định, YOLOv8 không có
  NMS trong đồ thị), True = nhúng NMS vào ONNX, False = dùng head end-to-end NMS-free nếu có
  (YOLO26/YOLOv10). Với YOLOv8 (không có head one2one), nms=False tương đương None: đồ thị không
  có NMS. Script vẫn kiểm tra lại để chắc chắn không còn NonMaxSuppression/TopK/... trong đồ thị.

Ví dụ:
  python optimization/export_onnx.py
  python optimization/export_onnx.py --weights models/best.pt --imgsz 640 --opset 11
"""

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import shape_inference
from onnx.utils import Extractor
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]

# Toán tử không thân thiện với NPU / bộ parse của Hailo DFC - không được xuất hiện trong đồ thị cắt
NPU_UNFRIENDLY_OPS = {"NonMaxSuppression", "TopK", "NonZero", "Loop", "If", "Scan", "GatherND",
                      "ScatterND", "Where", "Range", "RoiAlign"}
# Toán tử decode của Detect head - phải bị loại khỏi best_hailo.onnx
DECODE_OPS = {"Softmax", "Transpose", "Reshape", "Div", "Sub"}
# Dấu hiệu onnxslim thất bại (Ultralytics chỉ log cảnh báo): đồ thị shape tĩnh đã simplify không còn các nút này
UNSIMPLIFIED_OPS = {"Constant", "Shape", "Gather", "Unsqueeze", "ConstantOfShape"}


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #
def detect_head_info(yolo):
    """Lấy thông tin Detect head và tên 6 nút Conv cuối (quy ước đặt tên của Ultralytics)."""
    det = yolo.model
    idx = len(det.model) - 1
    head = det.model[idx]
    if head.reg_max <= 1 or getattr(det, "end2end", False):
        sys.exit("[LỖI] Script này dành cho Detect head kiểu YOLOv8 (DFL, reg_max=16, không end-to-end).")
    scales = range(len(head.cv2))
    return {
        "head_index": idx,
        "strides": [int(s) for s in head.stride],
        "reg_max": int(head.reg_max),
        "nc": int(head.nc),
        "names": {int(k): v for k, v in yolo.names.items()},
        # Thứ tự xen kẽ theo từng scale [reg_P3, cls_P3, reg_P4, cls_P4, reg_P5, cls_P5] - giống
        # exporter.export_hailo của Ultralytics và cấu hình yolov8 của Hailo Model Zoo
        "end_node_names": [f"/model.{idx}/cv{b}.{i}/cv{b}.{i}.2/Conv" for i in scales for b in (2, 3)],
    }


def tensor_shape(value_info):
    return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
            for d in value_info.type.tensor_type.shape.dim]


def op_histogram(model):
    return Counter(n.op_type for n in model.graph.node)


def letterbox(img, size):
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas


def load_sample_input(imgsz, sample):
    """Ảnh thật (letterbox, RGB, /255, NCHW) để kiểm tra số học; không có thì dùng nhiễu."""
    import cv2

    candidates = [Path(sample)] if sample else sorted(
        (REPO_ROOT / "datasets" / "processed" / "images" / "val").glob("*.jpg"))[:1]
    for path in candidates:
        img = cv2.imread(str(path))
        if img is not None:
            x = letterbox(img, imgsz)[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            return np.ascontiguousarray(x), path.name
    return np.random.default_rng(0).random((1, 3, imgsz, imgsz), dtype=np.float32), "random noise"


def decode_heads(outputs, info):
    """Decode 6 đầu ra thô [reg, cls] x scale (giống những gì CPU/Hailo phải làm) -> (1, 4 + nc, N)."""
    reg_max, boxes, scores = info["reg_max"], [], []
    for reg, cls, stride in zip(outputs[0::2], outputs[1::2], info["strides"]):
        _, _, h, w = reg.shape
        dist = reg.reshape(4, reg_max, h * w)
        dist = np.exp(dist - dist.max(1, keepdims=True))
        dist = (dist / dist.sum(1, keepdims=True) * np.arange(reg_max)[None, :, None]).sum(1)  # DFL -> (4, HW)
        sy, sx = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
        ax, ay = sx.reshape(-1), sy.reshape(-1)
        x1, y1, x2, y2 = ax - dist[0], ay - dist[1], ax + dist[2], ay + dist[3]
        boxes.append(np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]) * stride)
        scores.append(1.0 / (1.0 + np.exp(-cls.reshape(cls.shape[1], h * w))))
    return np.concatenate([np.concatenate(boxes, 1), np.concatenate(scores, 1)], 0)[None]


# --------------------------------------------------------------------------- #
# Các bước chính
# --------------------------------------------------------------------------- #
def export_full_onnx(weights, output, args):
    # Xuất từ bản sao tạm của .pt: Ultralytics luôn ghi <thư mục weights>/<stem>.onnx, xuất thẳng sẽ
    # ghi đè models/best.onnx khi người dùng chọn --output khác
    tmp_dir = Path(tempfile.mkdtemp(prefix="export_onnx_"))
    shutil.copy2(weights, tmp_dir / weights.name)
    try:
        exported = Path(YOLO(str(tmp_dir / weights.name)).export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,   # Hailo DFC 3.x parse ổn định nhất với opset 11 (Ultralytics cũng dùng 11 cho Hailo)
        simplify=True,      # onnxslim: gộp hằng số, loại nút thừa
        dynamic=False,      # NPU cần shape tĩnh
        half=False,         # FP32 - Hailo DFC tự lượng tử hóa INT8 từ đồ thị số thực
        batch=1,
        nms=False,          # không nhúng NMS
            device=args.device,
        ))
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), output)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return output


def verify_full_graph(path, imgsz):
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    inputs = model.graph.input
    if len(inputs) != 1 or tensor_shape(inputs[0]) != [1, 3, imgsz, imgsz]:
        sys.exit(f"[LỖI] Đầu vào phải là tĩnh [1, 3, {imgsz}, {imgsz}], nhận {[tensor_shape(i) for i in inputs]}")
    ops = set(op_histogram(model))
    if NPU_UNFRIENDLY_OPS & ops:
        sys.exit(f"[LỖI] Đồ thị đầy đủ còn chứa toán tử không phù hợp NPU: {sorted(NPU_UNFRIENDLY_OPS & ops)}")
    if UNSIMPLIFIED_OPS & ops:
        sys.exit(f"[LỖI] Đồ thị chưa được simplify (còn {sorted(UNSIMPLIFIED_OPS & ops)}) - onnxslim có thể đã lỗi, "
                 f"xem log phía trên (pip install -U onnxslim).")
    return model


def cut_for_hailo(full_path, model, info, output):
    nodes = {n.name: n for n in model.graph.node}
    missing = [e for e in info["end_node_names"] if e not in nodes]
    if missing:
        head_convs = sorted(n for n, node in nodes.items()
                            if node.op_type == "Conv" and n.startswith(f"/model.{info['head_index']}/"))
        sys.exit(f"[LỖI] Không tìm thấy end node {missing}.\nCác Conv của head hiện có: {head_convs}")

    out_tensors = [nodes[e].output[0] for e in info["end_node_names"]]
    inferred = shape_inference.infer_shapes(model)
    sub = Extractor(inferred).extract_model([model.graph.input[0].name], out_tensors)
    sub.producer_name = model.producer_name
    del sub.metadata_props[:]
    sub.metadata_props.extend(model.metadata_props)  # giữ stride/names/imgsz của Ultralytics
    onnx.checker.check_model(sub)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(sub, str(output))

    ops = op_histogram(sub)
    leftovers = (NPU_UNFRIENDLY_OPS | DECODE_OPS | UNSIMPLIFIED_OPS) & set(ops)
    if leftovers:
        sys.exit(f"[LỖI] Đồ thị cắt vẫn còn toán tử decode/không thân thiện NPU: {sorted(leftovers)}")
    outputs = [{"end_node": e, "tensor": t, "shape": tensor_shape(o),
                "kind": "reg" if "/cv2." in e else "cls"}
               for e, t, o in zip(info["end_node_names"], out_tensors, sub.graph.output)]
    return sub, ops, outputs


def numeric_check(full_path, hailo_path, info, imgsz, sample):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[BỎ QUA] Chưa cài onnxruntime - không kiểm tra số học (pip install onnxruntime).")
        return None
    x, src = load_sample_input(imgsz, sample)
    opts = {"providers": ["CPUExecutionProvider"]}
    full = ort.InferenceSession(str(full_path), **opts)
    cut = ort.InferenceSession(str(hailo_path), **opts)
    y_full = full.run(None, {full.get_inputs()[0].name: x})[0]
    y_cut = decode_heads(cut.run(None, {cut.get_inputs()[0].name: x}), info)
    box_err = float(np.abs(y_full[:, :4] - y_cut[:, :4]).max())
    cls_err = float(np.abs(y_full[:, 4:] - y_cut[:, 4:]).max())
    ok = y_full.shape == y_cut.shape and box_err < 1e-2 and cls_err < 1e-4
    print(f"\nKiểm tra số học ({src}): {full_path.name} {list(y_full.shape)} vs decode({hailo_path.name}) "
          f"{list(y_cut.shape)} | max|Δbox| = {box_err:.2e} px, max|Δscore| = {cls_err:.2e} -> "
          f"{'KHỚP' if ok else 'LỆCH'}")
    if not ok:
        sys.exit("[LỖI] Đồ thị cắt + decode trên CPU không tái tạo được đầu ra của đồ thị đầy đủ.")
    return {"sample": src, "max_abs_box_err_px": box_err, "max_abs_score_err": cls_err}


def parse_args():
    p = argparse.ArgumentParser(description="Xuất YOLOv8 sang ONNX và cắt đồ thị cho Hailo DFC.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", type=Path, default=REPO_ROOT / "models" / "best.pt")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "models" / "best.onnx")
    p.add_argument("--hailo-output", type=Path, help="Mặc định: <output>_hailo.onnx (vd models/best_hailo.onnx)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=11)
    p.add_argument("--device", type=str, default="cpu", help="Thiết bị dùng khi trace (cpu là đủ)")
    p.add_argument("--sample", type=str, help="Ảnh dùng để kiểm tra số học (mặc định: 1 ảnh tập val)")
    return p.parse_args()


def main():
    for stream in (sys.stdout, sys.stderr):  # tránh UnicodeEncodeError trên console Windows
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args()
    args.weights, args.output = args.weights.resolve(), args.output.resolve()
    args.hailo_output = (args.hailo_output or args.output.with_name(args.output.stem + "_hailo.onnx")).resolve()
    if args.imgsz % 32:
        sys.exit(f"[LỖI] imgsz={args.imgsz} phải là bội số của 32.")
    if not args.weights.is_file():
        sys.exit(f"[LỖI] Không tìm thấy {args.weights}. Chạy training/train.py trước.")

    info = detect_head_info(YOLO(str(args.weights)))
    full_path = export_full_onnx(args.weights, args.output, args)
    full_model = verify_full_graph(full_path, args.imgsz)
    sub, ops, outputs = cut_for_hailo(full_path, full_model, info, args.hailo_output)
    check = numeric_check(full_path, args.hailo_output, info, args.imgsz, args.sample)

    meta = {
        "weights": args.weights.name,
        "onnx_full": full_path.name,
        "onnx_hailo": args.hailo_output.name,
        "opset": args.opset,
        "input": {"name": full_model.graph.input[0].name, "shape": [1, 3, args.imgsz, args.imgsz],
                  "layout": "NCHW", "color": "RGB", "letterbox_pad": 114,
                  # Hailo nhận ảnh uint8 0-255 -> thêm lớp chuẩn hóa này trong model script của DFC
                  "hailo_normalization": {"mean": [0.0, 0.0, 0.0], "std": [255.0, 255.0, 255.0]}},
        "end_node_names": info["end_node_names"],
        "outputs": outputs,
        # Cặp (reg, cls) theo stride cho bbox_decoders trong nms_config.json của Hailo
        "bbox_decoders": [{"stride": st, "reg_node": info["end_node_names"][2 * i],
                           "cls_node": info["end_node_names"][2 * i + 1]} for i, st in enumerate(info["strides"])],
        "strides": info["strides"],
        "reg_max": info["reg_max"],
        "nc": info["nc"],
        "names": info["names"],
        "numeric_check": check,
    }
    meta_path = args.hailo_output.with_name(args.hailo_output.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    mb = lambda p: p.stat().st_size / 2**20  # noqa: E731
    print("\nToán tử trong best_hailo.onnx:", dict(sorted(ops.items())))
    print("\nĐầu ra của best_hailo.onnx (end nodes cho Hailo DFC):")
    for o in outputs:
        print(f"  [{o['kind']}] {o['end_node']:<34} {o['shape']}")
    print(f"\nKích thước: {args.weights.name} {mb(args.weights):.2f} MB | {full_path.name} {mb(full_path):.2f} MB | "
          f"{args.hailo_output.name} {mb(args.hailo_output):.2f} MB")
    print(f"Đã lưu: {full_path}\n        {args.hailo_output}\n        {meta_path}")


if __name__ == "__main__":
    main()
