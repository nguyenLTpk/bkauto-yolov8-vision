#!/usr/bin/env python3
"""
export_onnx.py - Giai đoạn 3.1: Xuất models/best.pt (YOLO26m, NMS-free) sang ONNX và chuẩn bị đồ thị cho Hailo DFC.

YOLO26 = head end-to-end NMS-free: nhánh one2one, KHÔNG có DFL (reg_max = 1).
  models/best.onnx         đồ thị đầy đủ, đầu ra (1, 300, 6) = [x1, y1, x2, y2, score, class]
                           (decode + sigmoid + TopK/GatherElements) -> baseline ONNX Runtime.
  models/best_hailo.onnx   cắt tại ĐÚNG 6 Conv cuối của nhánh one2one (one2one_cv2.{0,1,2} = box ltrb,
                           one2one_cv3.{0,1,2} = logit lớp): chỉ còn backbone + neck + head tích chập/attention.
                           TopK/Gather là phép phụ thuộc dữ liệu, không ánh xạ được lên dataflow của Hailo-8
                           -> decode ltrb*stride, sigmoid và top-300 chạy trên CPU (không cần NMS).
  models/best_hailo_meta.json
                           arch, 6 end_node_names, cặp (box, cls) theo stride, shape đầu ra, chuẩn hóa đầu vào
                           -> dùng cho compile_npu.py, edge_deployment/.

Về `nms=False` (Ultralytics 8.4): None = NMS ngoài; True = nhúng NMS; False = dùng head end-to-end NMS-free.
Với YOLO26, nms=False BẮT BUỘC để xuất nhánh one2one (Ultralytics khi đó bỏ hẳn nhánh one2many khỏi đồ thị).
Mô hình không phải YOLO26 (vd YOLOv8, có DFL và cần NMS) bị từ chối.

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

# Toán tử không ánh xạ được lên NPU / bộ parse của Hailo DFC - không được có trong đồ thị cắt
NPU_UNFRIENDLY_OPS = {"NonMaxSuppression", "TopK", "NonZero", "Loop", "If", "Scan", "GatherND", "GatherElements",
                      "ScatterND", "Where", "Range", "RoiAlign", "Mod", "ReduceMax", "ArgMax", "Cast"}
# Lưu ý: backbone YOLO26 có khối attention dùng hợp lệ Softmax/Reshape/Transpose/MatMul (Hailo-8 chạy được),
# nên các toán tử này KHÔNG bị cấm trong đồ thị cắt.
# Dấu hiệu onnxslim thất bại (Ultralytics chỉ log cảnh báo): đồ thị shape tĩnh đã simplify không còn các nút này
UNSIMPLIFIED_OPS = {"Constant", "Shape", "Gather", "Unsqueeze", "ConstantOfShape"}


# --------------------------------------------------------------------------- #
# Nhận diện head
# --------------------------------------------------------------------------- #
def detect_head_info(yolo):
    """Thông tin Detect head + 6 end-node Conv theo quy ước tên của Ultralytics."""
    det = yolo.model
    idx = len(det.model) - 1
    head = det.model[idx]
    if type(head).__name__ != "Detect":
        sys.exit(f"[LỖI] Chỉ hỗ trợ mô hình phát hiện (Detect head), nhận {type(head).__name__}.")
    has_one2one = getattr(head, "one2one_cv2", None) is not None
    if not (has_one2one and head.reg_max == 1):
        sys.exit(f"[LỖI] Không phải YOLO26 NMS-free (one2one={has_one2one}, reg_max={head.reg_max}). Pipeline chỉ hỗ trợ "
                 f"YOLO26: huấn luyện bằng training/configs/yolo26m_baseline.yaml.")
    arch, prefix = "yolo26", "one2one_"
    scales = range(head.nl)
    pairs = [{"stride": int(head.stride[i]),
              "reg_node": f"/model.{idx}/{prefix}cv2.{i}/{prefix}cv2.{i}.2/Conv",
              "cls_node": f"/model.{idx}/{prefix}cv3.{i}/{prefix}cv3.{i}.2/Conv"} for i in scales]
    # [reg_P3, reg_P4, reg_P5, cls_P3, cls_P4, cls_P5] - thứ tự exporter.export_hailo của Ultralytics
    end_nodes = [p["reg_node"] for p in pairs] + [p["cls_node"] for p in pairs]
    if len(end_nodes) != 6:
        sys.exit(f"[LỖI] Detect head có {head.nl} mức stride; pipeline cần đúng 3 (P3/P4/P5) -> 6 end-node.")
    return {
        "arch": arch,
        "end2end": True,
        "head_index": idx,
        "strides": [p["stride"] for p in pairs],
        "reg_max": int(head.reg_max),
        "nc": int(head.nc),
        "max_det": int(getattr(head, "max_det", 300)),
        "names": {int(k): v for k, v in yolo.names.items()},
        "pairs": pairs,
        "end_node_names": end_nodes,
    }


def tensor_shape(value_info):
    return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
            for d in value_info.type.tensor_type.shape.dim]


def op_histogram(model):
    return Counter(n.op_type for n in model.graph.node)


# --------------------------------------------------------------------------- #
# Tiền xử lý & decode tham chiếu (CPU) - cùng thuật toán với edge_deployment/
# --------------------------------------------------------------------------- #
def letterbox(img, size, pad=114):
    """Khớp ultralytics LetterBox(auto=False, center=True) - như calibrate_ptq.py và app trên Pi."""
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = round(w * r), round(h * r)
    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom, left, right = round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad, pad, pad))


def load_sample_input(imgsz, sample):
    """Ảnh thật (letterbox, RGB, /255, NCHW) để kiểm tra số học; không có thì dùng nhiễu."""
    import cv2

    candidates = [Path(sample)] if sample else sorted(
        (REPO_ROOT / "datasets" / "processed" / "images" / "val").glob("*.jpg"))[:1]
    for path in candidates:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            x = letterbox(img, imgsz)[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            return np.ascontiguousarray(x), path.name
    return np.random.default_rng(0).random((1, 3, imgsz, imgsz), dtype=np.float32), "random noise"


def _grid(h, w):
    sy, sx = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
    return sx.reshape(-1), sy.reshape(-1)


def decode_yolo26(scales, max_det=300):
    """scales: [(reg (1,4,H,W) khoảng cách ltrb theo stride, cls (1,nc,H,W) logit, stride)]
    -> (max_det, 6) [x1, y1, x2, y2, score, class] - tái hiện Detect._inference + Detect.postprocess:
       box = (anchor -/+ ltrb) * stride (không DFL), score = sigmoid, top-k 2 tầng (anchor rồi anchor x lớp)."""
    boxes, scores = [], []
    for reg, cls, stride in scales:
        _, _, h, w = reg.shape
        ltrb = reg.reshape(4, h * w)
        ax, ay = _grid(h, w)
        boxes.append(np.stack([ax - ltrb[0], ay - ltrb[1], ax + ltrb[2], ay + ltrb[3]], 1) * stride)
        scores.append((1.0 / (1.0 + np.exp(-cls.reshape(cls.shape[1], h * w)))).T)
    boxes, scores = np.concatenate(boxes), np.concatenate(scores)  # (N, 4), (N, nc)
    k = min(max_det, len(scores))
    anchors = np.argsort(-scores.max(1), kind="stable")[:k]
    flat = scores[anchors].reshape(-1)
    order = np.argsort(-flat, kind="stable")[:k]
    nc = scores.shape[1]
    chosen = anchors[order // nc]
    return np.concatenate([boxes[chosen], flat[order, None], (order % nc)[:, None].astype(np.float32)], 1)


def scales_from_outputs(outputs, meta_outputs, strides):
    """Ghép đầu ra (theo thứ tự end_node_names) thành [(reg, cls, stride)] theo đúng stride."""
    by_key = {(o["stride"], o["kind"]): arr for o, arr in zip(meta_outputs, outputs)}
    return [(by_key[(s, "reg")], by_key[(s, "cls")], s) for s in strides]


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
            half=False,         # FP32 - Hailo DFC tự lượng tử hóa từ đồ thị số thực
            batch=1,
            nms=False,          # BẮT BUỘC với YOLO26: xuất head end-to-end one2one (NMS-free)
            device=args.device,
        ))
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), output)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return output


def verify_full_graph(path, imgsz, info):
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    inputs, outputs = model.graph.input, model.graph.output
    if len(inputs) != 1 or tensor_shape(inputs[0]) != [1, 3, imgsz, imgsz]:
        sys.exit(f"[LỖI] Đầu vào phải là tĩnh [1, 3, {imgsz}, {imgsz}], nhận {[tensor_shape(i) for i in inputs]}")
    out_shape = tensor_shape(outputs[0])
    if out_shape != [1, info["max_det"], 6]:
        sys.exit(f"[LỖI] YOLO26 end-to-end phải xuất (1, {info['max_det']}, 6), nhận {out_shape}.")
    ops = set(op_histogram(model))
    if "NonMaxSuppression" in ops:
        sys.exit("[LỖI] Đồ thị đầy đủ không được chứa NonMaxSuppression.")
    # TopK/GatherElements/Unsqueeze... ở đuôi đồ thị đầy đủ là hợp lệ (chỉ dùng cho ONNX Runtime);
    # dấu hiệu simplify thất bại được kiểm tra đầy đủ trên đồ thị cắt.
    unsimplified = {"Shape", "ConstantOfShape"} & ops
    if unsimplified:
        sys.exit(f"[LỖI] Đồ thị chưa được simplify (còn {sorted(unsimplified)}) - onnxslim có thể đã lỗi, "
                 f"xem log phía trên (pip install -U onnxslim).")
    return model, out_shape


def cut_for_hailo(model, info, output):
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
    forbidden = NPU_UNFRIENDLY_OPS | UNSIMPLIFIED_OPS
    if forbidden & set(ops):
        sys.exit(f"[LỖI] Đồ thị cắt vẫn còn toán tử decode/không thân thiện NPU: {sorted(forbidden & set(ops))}")

    stride_of = {}
    for p in info["pairs"]:
        stride_of[p["reg_node"]], stride_of[p["cls_node"]] = (p["stride"], "reg"), (p["stride"], "cls")
    outputs = []
    for e, t, o in zip(info["end_node_names"], out_tensors, sub.graph.output):
        shape = tensor_shape(o)
        stride, kind = stride_of[e]
        want_c = 4 * info["reg_max"] if kind == "reg" else info["nc"]
        if shape[1] != want_c:
            sys.exit(f"[LỖI] {e} có {shape[1]} kênh, kỳ vọng {want_c} ({kind}).")
        outputs.append({"end_node": e, "tensor": t, "shape": shape, "kind": kind, "stride": stride})
    return ops, outputs


def match_end2end(a, b, score_tol=1e-5):
    """Ghép từng detection (x1, y1, x2, y2, score, cls) của a với b: cùng lớp, điểm lệch < score_tol, box gần nhất.
    Không so theo thứ tự hàng: các điểm (gần) bằng nhau có thể được torch.topk và numpy xếp khác nhau. Bỏ qua
    phần tử sát ngưỡng cắt top-k (hoà điểm ở biên có thể chọn anchor khác nhau)."""
    cutoff = max(a[:, 4].min(), b[:, 4].min()) + score_tol
    a, b = a[a[:, 4] > cutoff], b[b[:, 4] > cutoff]
    if not len(a):
        return 0.0, 0.0, 1.0
    used = np.zeros(len(b), dtype=bool)
    box_err = score_err = 0.0
    matched = 0
    for row in a:
        cand = np.nonzero(~used & (b[:, 5] == row[5]) & (np.abs(b[:, 4] - row[4]) < score_tol))[0]
        if not len(cand):
            continue
        dist = np.abs(b[cand, :4] - row[:4]).max(1)
        j = cand[dist.argmin()]
        used[j] = True
        matched += 1
        box_err = max(box_err, float(dist.min()))
        score_err = max(score_err, float(abs(b[j, 4] - row[4])))
    return box_err, score_err, matched / len(a)


def numeric_check(full_path, hailo_path, info, outputs, imgsz, sample):
    """So đồ thị đầy đủ với (đồ thị cắt + decode trên CPU) trên cùng một ảnh."""
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
    scales = scales_from_outputs(cut.run(None, {cut.get_inputs()[0].name: x}), outputs, info["strides"])

    y_cut = decode_yolo26(scales, info["max_det"])[None]
    box_err, cls_err, matched = match_end2end(y_full[0], y_cut[0])
    ok = y_full.shape == y_cut.shape and matched >= 0.99 and box_err < 1e-2 and cls_err < 1e-4
    print(f"  ghép cặp detection: {matched:.1%} khớp (cùng lớp, |Δscore| < 1e-5)")
    print(f"\nKiểm tra số học ({src}): {full_path.name} {list(y_full.shape)} vs decode({hailo_path.name}) "
          f"{list(y_cut.shape)} | max|Δbox| = {box_err:.2e} px, max|Δscore| = {cls_err:.2e} -> "
          f"{'KHỚP' if ok else 'LỆCH'}")
    if not ok:
        sys.exit("[LỖI] Đồ thị cắt + decode trên CPU không tái tạo được đầu ra của đồ thị đầy đủ.")
    return {"sample": src, "max_abs_box_err_px": box_err, "max_abs_score_err": cls_err}


def parse_args():
    p = argparse.ArgumentParser(description="Xuất YOLO26 (NMS-free) sang ONNX và cắt đồ thị cho Hailo DFC.",
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
    print(f"Kiến trúc: {info['arch']} | nc={info['nc']} | reg_max={info['reg_max']} | strides={info['strides']}"
          f" | end-to-end top-{info['max_det']} (NMS-free)")
    full_path = export_full_onnx(args.weights, args.output, args)
    full_model, full_out_shape = verify_full_graph(full_path, args.imgsz, info)
    ops, outputs = cut_for_hailo(full_model, info, args.hailo_output)
    check = numeric_check(full_path, args.hailo_output, info, outputs, args.imgsz, args.sample)

    meta = {
        "arch": info["arch"],
        "end2end": info["end2end"],
        "weights": args.weights.name,
        "onnx_full": full_path.name,
        "onnx_full_output": full_out_shape,
        "onnx_hailo": args.hailo_output.name,
        "opset": args.opset,
        "input": {"name": full_model.graph.input[0].name, "shape": [1, 3, args.imgsz, args.imgsz],
                  "layout": "NCHW", "color": "RGB", "letterbox_pad": 114,
                  # Hailo nhận ảnh uint8 0-255 -> thêm lớp chuẩn hóa này trong model script của DFC
                  "hailo_normalization": {"mean": [0.0, 0.0, 0.0], "std": [255.0, 255.0, 255.0]}},
        "end_node_names": info["end_node_names"],
        "outputs": outputs,
        "bbox_decoders": info["pairs"],  # cặp (reg, cls) theo stride
        "strides": info["strides"],
        "reg_max": info["reg_max"],
        "nc": info["nc"],
        "names": info["names"],
        "postprocess": {"type": "end2end_topk", "max_det": info["max_det"],
                        "box": "xyxy = (anchor -/+ ltrb) * stride, anchor = ô lưới + 0.5 (không DFL)",
                        "score": "sigmoid(cls logit) trên host", "nms": False},
        "numeric_check": check,
    }
    meta_path = args.hailo_output.with_name(args.hailo_output.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    mb = lambda p: p.stat().st_size / 2**20  # noqa: E731
    print(f"\nToán tử trong {args.hailo_output.name}:", dict(sorted(ops.items())))
    print(f"\nĐầu ra của {args.hailo_output.name} (end nodes cho Hailo DFC):")
    for o in outputs:
        print(f"  [{o['kind']} s{o['stride']:<2}] {o['end_node']:<44} {o['shape']}")
    print(f"\nKích thước: {args.weights.name} {mb(args.weights):.2f} MB | {full_path.name} {mb(full_path):.2f} MB | "
          f"{args.hailo_output.name} {mb(args.hailo_output):.2f} MB")
    print(f"Đã lưu: {full_path}\n        {args.hailo_output}\n        {meta_path}")


if __name__ == "__main__":
    main()
