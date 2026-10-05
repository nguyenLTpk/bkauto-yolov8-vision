#!/usr/bin/env python3
"""
compile_npu.py - Giai đoạn 3.3: Biên dịch models/best_hailo.onnx (YOLO26m, NMS-free) thành models/best_hailo.hef
cho Hailo-8 bằng Hailo Dataflow Compiler (hailo_sdk_client, DFC 3.x).

Môi trường: Linux x86_64 (Ubuntu 22.04/24.04) hoặc WSL2 trên Windows, Python 3.10-3.12, đã cài wheel
hailo_dataflow_compiler-3.x tải từ Hailo Developer Zone (xem optimization/requirements-hailo.txt).

YOLO26 là head end-to-end NMS-free, không DFL. HEF xuất thẳng 6 tensor thô của nhánh one2one (box ltrb 4 kênh +
logit lớp nc kênh, x 3 stride), độ chính xác a16 như exporter Hailo của Ultralytics. Script KHÔNG dùng
nms_postprocess và KHÔNG đặt sigmoid on-chip: TopK/Gather là phép phụ thuộc dữ liệu, không ánh xạ được lên
Hailo-8 -> decode + sigmoid + top-300 chạy trên CPU (edge_deployment/include/detection_utils.hpp).

Quy trình:
  [1/5] Parsing     : ONNX -> HN/HAR, cắt tại 6 end-node Conv one2one (đọc từ models/best_hailo_meta.json
                      do export_onnx.py sinh ra - BẮT BUỘC, phải có arch = yolo26).
  [2/5] Dò layer    : ánh xạ 6 output HN -> (stride, reg/cls) theo shape: C = 4 -> hồi quy, C = nc -> phân loại,
                      H -> stride.
  [3/5] Model script: normalization 0..255 trong NPU, calibset_size, PTQ INT8 (compression_level=0 -> không dùng
                      trọng số 4-bit), quantization_param(6 output, a16_w16).
  [4/5] Quantization: runner.optimize(tf.data phát từng ảnh calibration) -> HAR lượng tử hóa.
                      optimization_level tự chọn: 2 (+ fine-tune QFT) nếu có GPU, 1 nếu không.
  [5/5] Compilation : runner.compile() -> .hef + HAR đã biên dịch (dùng cho `hailo profiler`).

Ví dụ:
  python optimization/compile_npu.py
  python optimization/calibrate_ptq.py --num 1024 && python optimization/compile_npu.py --compiler-max  # server GPU

Đầu ra HEF ở dạng số nguyên lượng tử (uint16 với a16); HailoRT tự giải lượng tử sang FLOAT32 khi app đặt
output.set_format_type(HAILO_FORMAT_TYPE_FLOAT32) - DFC không cần cấu hình gì thêm.
"""

import argparse
import json
import platform
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def log(step, msg):
    print(f"[{step}] {msg}", flush=True)


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def __str__(self):
        return f"{time.perf_counter() - self.t0:.1f}s"


# --------------------------------------------------------------------------- #
# Đầu vào
# --------------------------------------------------------------------------- #
def load_export_meta(meta_path):
    """Đọc meta của export_onnx.py; chỉ chấp nhận YOLO26 NMS-free (nhánh one2one, reg_max = 1)."""
    if not meta_path.is_file():
        sys.exit(f"[LỖI] Không thấy {meta_path} - chạy optimization/export_onnx.py trước (file meta xác định "
                 f"6 end-node one2one của YOLO26).")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    nodes = meta.get("end_node_names", [])
    if meta.get("arch") != "yolo26" or meta.get("reg_max") != 1 or len(nodes) != 6 or \
            not all("/one2one_cv" in n for n in nodes):
        sys.exit(f"[LỖI] {meta_path.name} không phải YOLO26 NMS-free (arch={meta.get('arch')}, "
                 f"reg_max={meta.get('reg_max')}, end-node={nodes}). Pipeline chỉ hỗ trợ YOLO26: huấn luyện bằng "
                 f"training/configs/yolo26m_baseline.yaml rồi chạy lại export_onnx.py.")
    return {
        "max_det": meta.get("postprocess", {}).get("max_det", 300),
        "input_name": meta["input"]["name"],
        "input_shape": meta["input"]["shape"],
        "end_nodes": nodes,
        "strides": meta["strides"],
        "reg_max": meta["reg_max"],
        "nc": meta["nc"],
        "names": meta["names"],
    }


def load_calibration(calib_path, imgsz):
    """Mở calibration set dạng memmap (không nạp hết vào RAM) và kiểm tra tính nhất quán."""
    meta_path = calib_path.with_suffix(".json")
    data = np.load(calib_path, mmap_mode="r")
    if data.ndim != 4 or data.shape[1:] != (imgsz, imgsz, 3):
        sys.exit(f"[LỖI] Calibration set phải có shape (N, {imgsz}, {imgsz}, 3) NHWC, nhận {data.shape}. "
                 f"Chạy lại calibrate_ptq.py --imgsz {imgsz}.")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if list(meta.get("shape", [])) != list(data.shape) or len(meta.get("images", [])) != len(data):
            sys.exit(f"[LỖI] {meta_path.name} không khớp {calib_path.name} (shape {meta.get('shape')} vs "
                     f"{list(data.shape)}) - có thể do lần chạy calibrate_ptq.py bị lỗi. Chạy lại calibrate_ptq.py.")
    else:
        print(f"[CẢNH BÁO] Thiếu {meta_path.name} - suy ra cách chuẩn hóa từ dtype {data.dtype}.")
        meta = {}
    # uint8 0..255 -> cần lớp normalization trong NPU; float 0..1 -> đã chuẩn hóa trên host
    normalize_in_npu = meta.get("normalization_in_npu", data.dtype == np.uint8)
    peak, blank = 0.0, 0
    for start in range(0, len(data), 32):  # quét theo khối: giá trị lớn nhất + khung hình rỗng
        block = np.asarray(data[start:start + 32])
        peak = max(peak, float(block.max()))
        blank += int((block.reshape(len(block), -1).max(1) == 0).sum())
    if blank:
        sys.exit(f"[LỖI] {blank} ảnh calibration toàn số 0 (file ghi dở?). Chạy lại calibrate_ptq.py.")
    if normalize_in_npu and peak <= 1.0:
        sys.exit("[LỖI] Calibration set có giá trị <= 1 nhưng meta ghi chuẩn hóa trong NPU (0..255).")
    if not normalize_in_npu and peak > 1.0:
        sys.exit("[LỖI] Calibration set có giá trị > 1 nhưng meta ghi đã chuẩn hóa trên host (0..1).")
    return data, normalize_in_npu, meta


def calibration_feed(data):
    """Hàm tạo tf.data.Dataset phát từng ảnh float32 (như Hailo Model Zoo / Ultralytics truyền vào
    runner.optimize) - tránh bản sao float32 toàn bộ (~0.94 GB cho 200 ảnh, ~4.7 GB cho 1024 ảnh)."""
    import tensorflow as tf  # đi kèm DFC

    def generator():
        for image in data:
            yield np.asarray(image, dtype=np.float32), {}

    spec = (tf.TensorSpec(shape=data.shape[1:], dtype=tf.float32), {})
    return lambda: tf.data.Dataset.from_generator(generator, output_signature=spec)


# --------------------------------------------------------------------------- #
# Ánh xạ output HN -> nhánh hồi quy / phân loại theo stride
# --------------------------------------------------------------------------- #
def map_output_layers(runner, info, imgsz):
    """Trả về [{stride, reg_layer, cls_layer, reg_output, cls_output}] theo stride tăng dần."""
    if 4 * info["reg_max"] == info["nc"]:
        sys.exit(f"[LỖI] nc = {info['nc']} trùng số kênh hồi quy - không phân biệt được reg/cls theo shape.")
    found, out_names = {}, {}
    for out in runner.get_hn_model().get_output_layers():
        src = out.inputs[0].rsplit("/", 1)[-1]  # vd 'best_hailo/conv41' -> 'conv41'
        shape = [int(d) for d in out.output_shape if d not in (-1, None)]  # [H, W, C] (HN dùng NHWC)
        h, channels = shape[-3], shape[-1]
        kind = "reg" if channels == 4 * info["reg_max"] else "cls" if channels == info["nc"] else None
        if kind is None:
            sys.exit(f"[LỖI] Output {out.name} (từ {src}) có {channels} kênh, không khớp reg "
                     f"({4 * info['reg_max']}) hay cls ({info['nc']}). Kiểm tra end-node.")
        stride = imgsz // h
        if (stride, kind) in found:
            sys.exit(f"[LỖI] Hai output cùng stride {stride} loại {kind}: {found[(stride, kind)]}, {src}")
        found[(stride, kind)] = src
        out_names[(stride, kind)] = out.name.rsplit("/", 1)[-1]  # vd 'output_layer1' (tên dùng trong model script)
        print(f"      {out.name:<16} <- {src:<10} shape {shape} -> stride {stride:>2} {kind}")

    decoders = []
    for stride in sorted({s for s, _ in found}):
        if (stride, "reg") not in found or (stride, "cls") not in found:
            sys.exit(f"[LỖI] Thiếu cặp reg/cls cho stride {stride}: {found}")
        decoders.append({"stride": stride, "reg_layer": found[(stride, "reg")], "cls_layer": found[(stride, "cls")],
                         "reg_output": out_names[(stride, "reg")], "cls_output": out_names[(stride, "cls")]})
    if [d["stride"] for d in decoders] != sorted(info["strides"]):
        sys.exit(f"[LỖI] Stride dò được {[d['stride'] for d in decoders]} khác stride mô hình {info['strides']}")
    return decoders


def build_model_script(decoders, normalize_in_npu, calib_size, opt_level, args):
    lines = []
    if normalize_in_npu:
        # Lớp chuẩn hóa chạy trên NPU: (x - 0) / 255 -> Pi đẩy thẳng ảnh uint8 RGB vào HEF
        lines.append("normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])")
    lines += [
        f"model_optimization_config(calibration, batch_size={args.calib_batch}, calibset_size={calib_size})",
        f"model_optimization_flavor(optimization_level={opt_level}, compression_level=0)",
    ]
    if opt_level >= 2:  # fine-tune (QFT) trên đúng số ảnh đang có (mặc định DFC kỳ vọng 1024)
        lines.append(f"post_quantization_optimization(finetune, policy=enabled, dataset_size={calib_size})")
    # Đầu ra thô của head không DFL (khoảng cách ltrb + logit) có dải giá trị rộng: a8 làm mất độ chính xác box
    # -> giữ 16 bit ở 6 output (giống exporter Hailo của Ultralytics cho YOLO26). Không nms_postprocess, không
    # sigmoid on-chip: decode + top-k chạy trên host.
    outputs = ", ".join(n for d in decoders for n in (d["reg_output"], d["cls_output"]))
    lines.append(f"quantization_param([{outputs}], precision_mode=a16_w16)")
    if args.compiler_max:
        lines.append("performance_param(compiler_optimization_level=max)")
    return "\n".join(lines) + "\n"


def choose_opt_level(requested, calib_size):
    """optimization_level 2 (equalization + fine-tune QFT) cần GPU; không có GPU dùng 1 (equalization)."""
    if requested != "auto":
        return int(requested), "chỉ định qua --opt-level"
    try:
        import tensorflow as tf  # đi kèm DFC

        has_gpu = bool(tf.config.list_physical_devices("GPU"))
    except Exception:
        has_gpu = False
    if not has_gpu:
        return 1, "không có GPU (WSL2/CPU) - fine-tune QFT trên CPU quá chậm"
    note = "" if calib_size >= 1024 else f"; Hailo khuyến nghị >= 1024 ảnh (hiện {calib_size}: calibrate_ptq.py --num 1024)"
    return 2, f"có GPU{note}"


def hn_layer_summary(har_path):
    """Đọc file .hn (JSON) bên trong HAR (tar) để thống kê loại layer đã ánh xạ lên NPU."""
    try:
        with tarfile.open(har_path) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith(".hn"))
            hn = json.load(tar.extractfile(member))
        return dict(Counter(layer.get("type", "?") for layer in hn["layers"].values()))
    except Exception as e:  # định dạng HAR thay đổi giữa các bản DFC -> không chặn quy trình
        return {"error": f"không đọc được HN trong HAR: {e}"}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Biên dịch YOLO26 (NMS-free) ONNX -> HEF cho Hailo-8 bằng Hailo DFC.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--onnx", type=Path, default=REPO_ROOT / "models" / "best_hailo.onnx")
    p.add_argument("--meta", type=Path, help="Mặc định: <onnx>_meta.json do export_onnx.py sinh ra")
    p.add_argument("--calib", type=Path, default=REPO_ROOT / "models" / "calib" / "calib_set_640.npy")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "models" / "best_hailo.hef")
    p.add_argument("--hw-arch", default="hailo8", choices=("hailo8", "hailo8l", "hailo8r"),
                   help="hailo8 = Raspberry Pi AI HAT+ 26 TOPS / M.2 Hailo-8")
    p.add_argument("--opt-level", default="auto", choices=("auto", "0", "1", "2"),
                   help="0 = chỉ PTQ, 1 = + equalization, 2 = + fine-tune QFT (cần GPU); auto theo GPU")
    p.add_argument("--calib-batch", type=int, default=8)
    p.add_argument("--compiler-max", action="store_true",
                   help="performance_param(compiler_optimization_level=max): FPS cao hơn, biên dịch lâu hơn nhiều")
    args = p.parse_args()
    args.onnx = args.onnx.resolve()
    args.meta = (args.meta or args.onnx.with_name(args.onnx.stem + "_meta.json")).resolve()
    args.calib, args.output = args.calib.resolve(), args.output.resolve()
    return args


def main():
    for stream in (sys.stdout, sys.stderr):  # tránh UnicodeEncodeError trên console Windows/WSL
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args()
    if platform.system() != "Linux":
        sys.exit("[LỖI] Hailo DFC chỉ chạy trên Linux x86_64. Trên Windows hãy chạy script này trong WSL2 "
                 "(Ubuntu 22.04) - xem optimization/requirements-hailo.txt.")
    try:
        import hailo_sdk_client
        from hailo_sdk_client import ClientRunner
    except ImportError:
        sys.exit("[LỖI] Chưa cài Hailo Dataflow Compiler (hailo_sdk_client). "
                 "Tải wheel hailo_dataflow_compiler-3.x từ https://hailo.ai/developer-zone/ rồi pip install.")
    for path, hint in ((args.onnx, "optimization/export_onnx.py"), (args.calib, "optimization/calibrate_ptq.py")):
        if not path.is_file():
            sys.exit(f"[LỖI] Không tìm thấy {path}. Chạy {hint} trước.")

    total = Timer()
    net_name = args.onnx.stem
    build_dir = args.output.parent / f"{args.output.stem}_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    info = load_export_meta(args.meta)
    _, _, imgsz, imgsz_w = info["input_shape"]
    if imgsz != imgsz_w or imgsz % 32:
        sys.exit(f"[LỖI] Đầu vào {info['input_shape']} phải vuông và là bội số của 32.")
    dfc_version = getattr(hailo_sdk_client, "__version__", "?")
    print(f"Hailo DFC {dfc_version} | hw_arch={args.hw_arch} | YOLO26 NMS-free | input {info['input_name']} "
          f"{info['input_shape']} | nc={info['nc']} | thư mục build: {build_dir}")
    print("      HEF xuất 6 tensor thô (a16) - decode + sigmoid + top-k trên host, không nms_postprocess.")

    # ---- [1/5] Parsing ----
    t = Timer()
    log("1/5", f"Parsing {args.onnx.name}, end-nodes:")
    for node in info["end_nodes"]:
        print(f"      {node}")
    runner = ClientRunner(hw_arch=args.hw_arch)
    runner.translate_onnx_model(
        str(args.onnx),
        net_name,
        start_node_names=[info["input_name"]],
        end_node_names=info["end_nodes"],
        net_input_shapes={info["input_name"]: info["input_shape"]},
    )
    parsed_har = build_dir / f"{net_name}_parsed.har"
    runner.save_har(str(parsed_har))
    log("1/5", f"Xong ({t}) -> {parsed_har.name}")

    # ---- [2/5] Ánh xạ output ----
    log("2/5", "Ánh xạ output HN -> nhánh hồi quy (reg) / phân loại (cls) theo stride:")
    decoders = map_output_layers(runner, info, imgsz)

    # ---- [3/5] Model script ----
    calib, normalize_in_npu, calib_meta = load_calibration(args.calib, imgsz)
    opt_level, opt_reason = choose_opt_level(args.opt_level, len(calib))
    (build_dir / "nms_config.json").unlink(missing_ok=True)  # tàn dư của các lần biên dịch YOLOv8 cũ (nếu có)
    script = build_model_script(decoders, normalize_in_npu, len(calib), opt_level, args)
    script_path = build_dir / f"{net_name}.alls"
    script_path.write_text(script, encoding="utf-8")
    log("3/5", f"Model script ({script_path.name}), optimization_level={opt_level} ({opt_reason}):")
    print("      " + script.strip().replace("\n", "\n      "))
    runner.load_model_script(script)

    # ---- [4/5] Quantization ----
    t = Timer()
    log("4/5", f"PTQ INT8 với {len(calib)} ảnh calibration {calib.shape} "
               f"({'0..255, chuẩn hóa trong NPU' if normalize_in_npu else '0..1'}) - có thể mất vài phút ...")
    n_calib = len(calib)
    runner.optimize(calibration_feed(calib))
    quant_har = build_dir / f"{net_name}_quantized.har"
    runner.save_har(str(quant_har))
    log("4/5", f"Xong ({t}) -> {quant_har.name}")
    del calib

    # ---- [5/5] Compilation ----
    t = Timer()
    log("5/5", f"Biên dịch cho {args.hw_arch}{' (compiler_optimization_level=max)' if args.compiler_max else ''} ...")
    hef = runner.compile()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(hef)
    compiled_har = build_dir / f"{net_name}_compiled.har"
    runner.save_har(str(compiled_har))
    log("5/5", f"Xong ({t}) -> {args.output.name} ({args.output.stat().st_size / 2**20:.2f} MB)")

    # ---- Báo cáo ----
    layer_types = hn_layer_summary(compiled_har)
    sizes = {p.name: round(p.stat().st_size / 2**20, 3)
             for p in (REPO_ROOT / "models" / "best.pt", REPO_ROOT / "models" / "best.onnx", args.onnx, args.output)
             if p.is_file()}
    report = {
        "dfc_version": dfc_version,
        "hw_arch": args.hw_arch,
        "onnx": args.onnx.name,
        "hef": args.output.name,
        "input": {"name": info["input_name"], "shape_nchw": info["input_shape"], "hef_input": "uint8 NHWC RGB"
                  if normalize_in_npu else "float NHWC RGB 0..1", "letterbox_pad": calib_meta.get("letterbox_pad")},
        "end_nodes": info["end_nodes"],
        "bbox_decoders": decoders,
        "arch": "yolo26",
        "classes": info["names"],
        "postprocess": ("end-to-end NMS-free: 6 raw outputs (a16) -> host decode xyxy=(anchor -/+ ltrb)*stride, "
                        f"sigmoid, top-{info['max_det']}"),
        "output_format_runtime": "HailoRT dequantize -> FLOAT32 NHWC (output.set_format_type(HAILO_FORMAT_TYPE_FLOAT32))",
        "quantization": {"precision": "INT8 (a8_w8), compression_level=0, output layers a16_w16",
                         "optimization_level": opt_level,
                         "reason": opt_reason, "calibration_images": n_calib,
                         "calibration_file": args.calib.name},
        "model_script": script.splitlines(),
        "hn_layer_types": layer_types,
        "file_sizes_mb": sizes,
        "elapsed_s": round(time.perf_counter() - total.t0, 1),
    }
    report_path = build_dir / "compile_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nHoàn tất trong {total}!")
    print(f"  HEF           : {args.output}")
    print(f"  Báo cáo       : {report_path}")
    print(f"  Kích thước MB : {sizes}")
    print(f"  Layer trên NPU: {layer_types}")
    print("  Ghi chú: mọi layer trong HEF chạy trên NPU Hailo-8 (DFC không có CPU fallback - toán tử không hỗ trợ "
          "sẽ lỗi ngay ở bước parsing); decode + sigmoid + top-k (NMS-free) chạy trên CPU của Pi.")
    print(f"  Phân tích chi tiết từng layer: hailo profiler {compiled_har}")


if __name__ == "__main__":
    main()
