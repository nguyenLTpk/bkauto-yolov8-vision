#!/usr/bin/env python3
"""
benchmark_profiler.py - Giai đoạn 4: Đo kiểm tự động (headless) trên Raspberry Pi 5 + Hailo-8.

Lệnh con:
  accuracy  Chạy toàn bộ tập Test qua mô hình, tính mAP@0.5, mAP@0.5:0.95, P, R, F1 (toàn cục + từng lớp).
            Thuật toán khớp & AP được port nguyên văn từ Ultralytics (ap_per_class / compute_ap /
            match_predictions, nội suy 101 điểm COCO) sang numpy -> so sánh trực tiếp với mAP baseline.
            --backend hailo : HEF YOLO26 qua HailoRT (InferVStreams, 6 đầu ra thô FLOAT32) + decode
                              end-to-end NMS-free trên CPU (sigmoid + top-k, cùng thuật toán với app C++)
            --backend onnx  : best.onnx qua onnxruntime (đầu ra YOLO26 end-to-end (1, 300, 6), không NMS),
                              CÙNG tiền xử lý
                              -> ΔmAP = mAP(NPU) - mAP(ONNX FP32) chỉ còn phản ánh sai số lượng tử hóa.
  npu       `hailortcli benchmark`: FPS hw-only/streaming và độ trễ phần cứng thuần của NPU.
  stress    Chạy suy luận liên tục (mặc định 300 s) và lấy mẫu mỗi giây: RAM (MB), CPU (%), nhiệt độ SoC
            (/sys/class/thermal/thermal_zone0/temp), nhiệt độ NPU, tần số CPU, throttling.
            --engine python : vòng lặp HailoRT Python trong chính tiến trình này
            --engine cpp    : chạy ./build/inference_app (headless) và theo dõi tiến trình đó - đo đúng
                              ứng dụng triển khai thật; độ trễ từng tầng đọc từ file --stats của app.
  all       accuracy (+ baseline ONNX nếu có --onnx) + npu + stress -> bảng tổng hợp summary.csv/json
            theo đúng các nhóm chỉ số của đề bài.

Ví dụ (chạy từ gốc repo trên Pi):
  python3 edge_deployment/benchmark_profiler.py accuracy --hef models/best_hailo.hef
  python3 edge_deployment/benchmark_profiler.py stress --engine cpp --source test_video.mp4
  python3 edge_deployment/benchmark_profiler.py all --hef models/best_hailo.hef --onnx models/best.onnx
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import psutil
except ImportError:  # psutil chỉ bắt buộc cho lệnh stress
    psutil = None

REPO_ROOT = Path(__file__).resolve().parents[1]
NAMES = {0: "traffic_sign", 1: "pedestrian"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PAD_VALUE = 114
IOUV = np.linspace(0.5, 0.95, 10)
SOC_TEMP_PATH = Path("/sys/class/thermal/thermal_zone0/temp")
LOG = logging.getLogger("benchmark")


# =============================================================================================== #
# Tiện ích
# =============================================================================================== #
def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "benchmark.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        LOG.addHandler(handler)


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    LOG.info("Đã ghi %s", path)


def imread(path: Path):
    """cv2.imread không mở được đường dẫn Unicode trên Windows - dùng imdecode cho mọi nền tảng."""
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def letterbox(img: np.ndarray, size: int, pad: int = PAD_VALUE):
    """Khớp ultralytics LetterBox(auto=False, center=True), calibrate_ptq.py và inference_app.cpp."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = round(w * r), round(h * r)
    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom, left, right = round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)
    out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad, pad, pad))
    return out, r, (left, top)


def preprocess(bgr: np.ndarray, size: int):
    img, r, pad = letterbox(bgr, size)
    return np.ascontiguousarray(img[:, :, ::-1]), r, pad  # RGB, HWC uint8


def scale_boxes_to_original(boxes: np.ndarray, r: float, pad, shape, clip: bool = True) -> np.ndarray:
    """xyxy trên ảnh model (pixel) -> xyxy trên ảnh gốc. clip=False khi đo mAP: Ultralytics val tính IoU trên
    box chưa cắt biên, cắt biên làm mAP@0.5:0.95 cao hơn thực tế với box sát mép ảnh."""
    out = boxes.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / r
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / r
    if clip:
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, shape[1])
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, shape[0])
    return out


def read_soc_temp() -> float | None:
    try:
        return int(SOC_TEMP_PATH.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def read_throttled() -> str | None:
    """vcgencmd get_throttled: 0x0 = bình thường; bit 0/2 = đang giảm áp/đang throttle."""
    if not shutil.which("vcgencmd"):
        return None
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
        return out.strip().split("=")[-1] or None
    except (OSError, subprocess.SubprocessError):
        return None


def system_info() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "machine": platform.machine()}
    model = Path("/proc/device-tree/model")
    if model.is_file():
        info["board"] = model.read_text(errors="ignore").strip("\x00\n ")
    if shutil.which("hailortcli"):
        try:
            info["hailortcli"] = subprocess.run(["hailortcli", "--version"], capture_output=True, text=True,
                                                timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def file_sizes() -> dict:
    sizes = {}
    for name in ("best.pt", "best.onnx", "best_hailo.onnx", "best_hailo.hef"):
        p = REPO_ROOT / "models" / name
        if p.is_file():
            sizes[name] = round(p.stat().st_size / 2**20, 3)
    return sizes


# =============================================================================================== #
# Dữ liệu Test
# =============================================================================================== #
def load_dataset(images_dir: Path, labels_dir: Path, limit: int = 0):
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"[LỖI] Không có ảnh trong {images_dir}")
    samples = []
    for p in paths:
        lbl = labels_dir / (p.stem + ".txt")
        rows = []
        if lbl.is_file():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 5:
                    rows.append([float(v) for v in parts])
        samples.append((p, np.array(rows, dtype=np.float32).reshape(-1, 5)))
    return samples


def labels_to_xyxy(labels: np.ndarray, w: int, h: int):
    cls = labels[:, 0].astype(int)
    xc, yc, bw, bh = labels[:, 1] * w, labels[:, 2] * h, labels[:, 3] * w, labels[:, 4] * h
    return np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], 1), cls


# =============================================================================================== #
# Backend suy luận: infer(rgb uint8 HWC) -> đầu ra thô;  postprocess(thô) -> (N, 6)
#   [x1, y1, x2, y2 (pixel ảnh model 640x640), score, cls]. Tách 2 bước để đo riêng thời gian hậu xử lý.
# =============================================================================================== #
def decode_end2end(scales, score_th: float, max_det: int) -> np.ndarray:
    """Hậu xử lý YOLO26 end-to-end (NMS-free) - GIỐNG HỆT edge::decode_end2end_model() trong
    edge_deployment/include/detection_utils.hpp (cùng thứ tự, cùng lớp, box và score trùng từng bit).
    Tương đương ultralytics Detect.postprocess (top-k 2 tầng).

    scales: [(box (H, W, 4) khoảng cách ltrb theo stride, cls (H, W, nc) logit, stride)] layout NHWC, float32.
      1. Ngưỡng: logit > float32(log(th / (1 - th))) (so sánh float32; th <= 0 -> mọi ô, th >= 1 -> không ô nào).
      2. Ứng viên = cặp (anchor, lớp) vượt ngưỡng theo thứ tự np.nonzero (stride, ô theo hàng, lớp).
      3. Sắp xếp giảm dần theo LOGIT, bằng nhau giữ thứ tự bước 2 (stable); giữ max_det phần tử đầu.
      4. box xyxy = (anchor -/+ ltrb) * stride, anchor = ô + 0.5 (float32, không DFL);
         score = float32(sigmoid(logit)) tính bằng double.
    Trả về (K, 6) [x1, y1, x2, y2 (pixel khung model), score, cls] float32."""
    logits, boxes = [], []
    for box, cls, stride in scales:
        h, w = box.shape[:2]
        sy, sx = np.meshgrid(np.arange(h, dtype=np.float32) + np.float32(0.5),
                             np.arange(w, dtype=np.float32) + np.float32(0.5), indexing="ij")
        ltrb = box.reshape(-1, 4).astype(np.float32, copy=False)
        boxes.append((sx.reshape(-1), sy.reshape(-1), ltrb, np.float32(stride)))
        logits.append(cls.reshape(-1, cls.shape[-1]).astype(np.float32, copy=False))
    logits = np.concatenate(logits)
    if score_th <= 0:
        logit_th = np.float32(-np.inf)
    elif score_th >= 1:
        logit_th = np.float32(np.inf)
    else:
        logit_th = np.float32(np.log(score_th / (1.0 - score_th)))
    anchor, cls_id = np.nonzero(logits > logit_th)
    if not len(anchor):
        return np.zeros((0, 6), np.float32)
    cand = logits[anchor, cls_id]
    order = np.argsort(-cand, kind="stable")[:max_det]
    anchor, cls_id, cand = anchor[order], cls_id[order], cand[order]

    ax = np.concatenate([b[0] for b in boxes])[anchor]
    ay = np.concatenate([b[1] for b in boxes])[anchor]
    ltrb = np.concatenate([b[2] for b in boxes])[anchor]
    st = np.concatenate([np.full(len(b[0]), b[3], np.float32) for b in boxes])[anchor]
    xyxy = np.stack([(ax - ltrb[:, 0]) * st, (ay - ltrb[:, 1]) * st, (ax + ltrb[:, 2]) * st, (ay + ltrb[:, 3]) * st], 1)
    score = (1.0 / (1.0 + np.exp(-cand.astype(np.float64)))).astype(np.float32)  # double -> float32 (như C++)
    return np.concatenate([xyxy, score[:, None], cls_id[:, None].astype(np.float32)], 1).astype(np.float32)


class HailoBackend:
    """HEF YOLO26 NMS-free qua InferVStreams (API ổn định trên HailoRT 4.x): 6 đầu ra thô FLOAT32 NHWC
    (HailoRT giải lượng tử a16 -> float) = box (H, W, 4) + logit lớp (H, W, nc) cho stride 8/16/32."""

    name = "hailo"

    def __init__(self, hef_path: Path, score_th: float, max_det: int):
        from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm,
                                    HailoStreamInterface, InferVStreams, InputVStreamParams,
                                    OutputVStreamParams, VDevice)

        self.hef = HEF(str(hef_path))
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.size = int(self.input_info.shape[0])
        self.score_th, self.max_det = score_th, max_det
        self.scales, self.nc = self._pair_outputs(self.hef.get_output_vstream_infos(), self.size)
        self._stack = ExitStack()
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE  # tự activate, không qua scheduler
        self.vdevice = self._stack.enter_context(VDevice(params))
        cfg = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        network_group = self.vdevice.configure(self.hef, cfg)[0]
        in_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        self.pipeline = self._stack.enter_context(InferVStreams(network_group, in_params, out_params))
        self._stack.enter_context(network_group.activate(network_group.create_params()))
        self._device = None
        try:
            self._device = self.vdevice.get_physical_devices()[0]
        except Exception as e:  # không chặn đo kiểm nếu không đọc được cảm biến
            LOG.warning("Không truy cập được thiết bị vật lý để đọc nhiệt độ NPU: %s", e)

    @staticmethod
    def _pair_outputs(infos, size: int):
        """Ghép (box 4 kênh, cls nc kênh) theo stride = cạnh đầu vào / chiều cao feature map."""
        if len(infos) != 6:
            raise SystemExit(f"[LỖI] HEF có {len(infos)} đầu ra; YOLO26 NMS-free cần 6 (box + cls x 3 stride). "
                             f"Xuất lại bằng optimization/export_onnx.py rồi compile_npu.py.")
        found, ncs = {}, set()
        for info in infos:
            h, _, c = (int(v) for v in info.shape)
            kind = "box" if c == 4 else "cls"
            if kind == "cls":
                ncs.add(c)
            found[(size // h, kind)] = info.name
        strides = sorted({s for s, _ in found})
        if len(ncs) != 1 or len(strides) != 3 or any((s, k) not in found for s in strides for k in ("box", "cls")):
            raise SystemExit(f"[LỖI] Không ghép được 3 cặp (box, cls) từ đầu ra HEF: {sorted(found)}")
        return [(found[(s, "box")], found[(s, "cls")], s) for s in strides], ncs.pop()

    def infer(self, rgb: np.ndarray):
        return self.pipeline.infer({self.input_info.name: rgb[None]})  # {tên: (1, H, W, C) float32}

    def postprocess(self, raw) -> np.ndarray:
        return decode_end2end([(raw[b][0], raw[c][0], s) for b, c, s in self.scales], self.score_th, self.max_det)

    def npu_temperature(self) -> float | None:
        if self._device is None:
            return None
        try:
            return float(self._device.control.get_chip_temperature().ts0_temperature)
        except Exception:
            return None

    def close(self):
        self._stack.close()


class OnnxBackend:
    """Baseline FP32 bằng onnxruntime trên models/best.onnx (đồ thị đầy đủ YOLO26 của export_onnx.py): đầu ra
    (1, 300, 6) [x1, y1, x2, y2, score, cls] đã decode + top-k end-to-end - chỉ lọc theo ngưỡng, KHÔNG NMS."""

    name = "onnx"

    def __init__(self, onnx_path: Path, score_th: float):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp, out = self.session.get_inputs()[0], self.session.get_outputs()[0]
        if len(out.shape) != 3 or out.shape[-1] != 6:
            raise SystemExit(f"[LỖI] {onnx_path.name} có đầu ra {out.shape}, không phải YOLO26 end-to-end (1, 300, 6). "
                             f"Xuất lại bằng optimization/export_onnx.py.")
        self.input_name, self.size = inp.name, int(inp.shape[2])
        self.score_th = score_th

    def infer(self, rgb: np.ndarray):
        x = (rgb.transpose(2, 0, 1)[None].astype(np.float32)) / 255.0
        return self.session.run(None, {self.input_name: x})[0][0]

    def postprocess(self, pred) -> np.ndarray:
        return pred[pred[:, 4] > self.score_th].astype(np.float32)

    def npu_temperature(self):
        return None

    def close(self):
        pass


def make_backend(args, score_th: float):
    if args.backend == "onnx":
        return OnnxBackend(args.onnx, score_th)
    return HailoBackend(args.hef, score_th, args.max_det)


# =============================================================================================== #
# Metric - port numpy của ultralytics/utils/metrics.py (ap_per_class, compute_ap, smooth) và
# engine/validator.py (match_predictions) để mAP trên Pi so sánh trực tiếp với mAP baseline
# =============================================================================================== #
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    (a1, a2), (b1, b2) = np.split(a[:, None], 2, axis=2), np.split(b[None], 2, axis=2)
    inter = (np.minimum(a2, b2) - np.maximum(a1, b1)).clip(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + 1e-7)


def match_predictions(pred_cls: np.ndarray, true_cls: np.ndarray, iou: np.ndarray) -> np.ndarray:
    correct = np.zeros((pred_cls.shape[0], len(IOUV)), dtype=bool)
    iou = iou * (true_cls[:, None] == pred_cls)
    for i, threshold in enumerate(IOUV):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(np.interp(x, mrec, mpre), x))


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    x = np.linspace(0, 1, 1000)
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l, n_p = nt[ci], i.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc, tpc = (1 - tp[i]).cumsum(0), tp[i].cumsum(0)
        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    idx = smooth(f1_curve.mean(0), 0.1).argmax()
    return p_curve[:, idx], r_curve[:, idx], f1_curve[:, idx], ap, unique_classes.astype(int), float(x[idx]), nt


def compute_metrics(stats: dict) -> dict:
    tp = np.concatenate(stats["tp"]) if stats["tp"] else np.zeros((0, len(IOUV)), bool)
    conf = np.concatenate(stats["conf"]) if stats["conf"] else np.zeros(0)
    pred_cls = np.concatenate(stats["pred_cls"]) if stats["pred_cls"] else np.zeros(0)
    target_cls = np.concatenate(stats["target_cls"]) if stats["target_cls"] else np.zeros(0)
    if not len(target_cls):
        raise SystemExit("[LỖI] Tập test không có nhãn nào")
    p, r, f1, ap, classes, best_conf, nt = ap_per_class(tp, conf, pred_cls, target_cls)
    per_class = {}
    for k, c in enumerate(classes):
        per_class[NAMES.get(int(c), str(c))] = {
            "instances": int(nt[k]), "precision": float(p[k]), "recall": float(r[k]), "f1": float(f1[k]),
            "mAP50": float(ap[k, 0]), "mAP50_95": float(ap[k].mean())}
    return {
        "precision": float(p.mean()), "recall": float(r.mean()), "f1": float(f1.mean()),
        "mAP50": float(ap[:, 0].mean()), "mAP50_95": float(ap.mean()),
        "best_f1_conf": best_conf, "per_class": per_class,
    }


# =============================================================================================== #
# Lệnh: accuracy
# =============================================================================================== #
def run_accuracy(args, out_dir: Path, backend_name: str | None = None) -> dict:
    backend_name = backend_name or args.backend
    args_backend, args.backend = args.backend, backend_name
    images_dir = args.dataset / "images" / args.split
    labels_dir = args.dataset / "labels" / args.split
    samples = load_dataset(images_dir, labels_dir, args.limit)
    LOG.info("[accuracy/%s] %d ảnh từ %s (conf=%.3f, max_det=%d)", backend_name, len(samples), images_dir,
             args.conf, args.max_det)
    backend = make_backend(args, args.conf)
    args.backend = args_backend
    stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
    t_pre, t_inf, t_post = [], [], []
    try:
        for k, (path, labels) in enumerate(samples, 1):
            bgr = imread(path)
            if bgr is None:
                LOG.warning("Bỏ qua ảnh không đọc được: %s", path.name)
                continue
            h, w = bgr.shape[:2]
            t0 = time.perf_counter()
            rgb, r, pad = preprocess(bgr, backend.size)
            t1 = time.perf_counter()
            raw = backend.infer(rgb)
            t2 = time.perf_counter()
            pred = backend.postprocess(raw)  # YOLO26: decode + sigmoid + top-k (NMS-free) trên CPU
            boxes = scale_boxes_to_original(pred[:, :4], r, pad, (h, w), clip=False)
            t3 = time.perf_counter()
            t_pre.append((t1 - t0) * 1e3), t_inf.append((t2 - t1) * 1e3), t_post.append((t3 - t2) * 1e3)

            gt_boxes, gt_cls = labels_to_xyxy(labels, w, h)
            stats["target_cls"].append(gt_cls)
            stats["conf"].append(pred[:, 4])
            stats["pred_cls"].append(pred[:, 5].astype(int))
            if len(pred) == 0:
                stats["tp"].append(np.zeros((0, len(IOUV)), bool))
            elif len(gt_cls) == 0:
                stats["tp"].append(np.zeros((len(pred), len(IOUV)), bool))
            else:
                stats["tp"].append(match_predictions(pred[:, 5].astype(int), gt_cls, box_iou(gt_boxes, boxes)))
            if k % 200 == 0 or k == len(samples):
                LOG.info("  %d/%d ảnh", k, len(samples))
    finally:
        backend.close()

    metrics = compute_metrics(stats)
    metrics.update({
        "backend": backend_name, "model": str(args.hef if backend_name == "hailo" else args.onnx),
        "images": len(t_inf), "conf_threshold": args.conf, "max_det": args.max_det,
        "timing_ms": {"preprocess": float(np.mean(t_pre)), "inference": float(np.mean(t_inf)),
                      "postprocess": float(np.mean(t_post)),
                      "inference_p95": float(np.percentile(t_inf, 95))},
    })
    LOG.info("[accuracy/%s] mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f  F1=%.4f", backend_name, metrics["mAP50"],
             metrics["mAP50_95"], metrics["precision"], metrics["recall"], metrics["f1"])
    for name, m in metrics["per_class"].items():
        LOG.info("    %-13s n=%-5d P=%.4f R=%.4f mAP50=%.4f mAP50-95=%.4f", name, m["instances"], m["precision"],
                 m["recall"], m["mAP50"], m["mAP50_95"])
    save_json(out_dir / f"accuracy_{backend_name}.json", metrics)
    with open(out_dir / f"accuracy_{backend_name}_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "instances", "precision", "recall", "f1", "mAP50", "mAP50_95"])
        w.writerow(["all", sum(m["instances"] for m in metrics["per_class"].values()), metrics["precision"],
                    metrics["recall"], metrics["f1"], metrics["mAP50"], metrics["mAP50_95"]])
        for name, m in metrics["per_class"].items():
            w.writerow([name, m["instances"], m["precision"], m["recall"], m["f1"], m["mAP50"], m["mAP50_95"]])
    return metrics


# =============================================================================================== #
# Lệnh: npu (hailortcli benchmark)
# =============================================================================================== #
def run_npu_benchmark(args, out_dir: Path) -> dict | None:
    if not shutil.which("hailortcli"):
        LOG.warning("[npu] Không có hailortcli - bỏ qua")
        return None
    cmd = ["hailortcli", "benchmark", str(args.hef), "-t", str(args.npu_seconds)]
    LOG.info("[npu] %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.npu_seconds * 6 + 120)
    (out_dir / "npu_benchmark.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        LOG.warning("[npu] hailortcli lỗi (mã %d), xem npu_benchmark.txt", proc.returncode)
        return None

    def grab(pattern):
        m = re.search(pattern, proc.stdout)
        return float(m.group(1)) if m else None

    result = {
        "fps_hw_only": grab(r"FPS\s+\(hw_only\)\s+=\s+([\d.]+)"),
        "fps_streaming": grab(r"\(streaming\)\s+=\s+([\d.]+)"),
        "latency_hw_ms": grab(r"Latency\s+\(hw\)\s+=\s+([\d.]+)"),
        "latency_overall_ms": grab(r"\(overall\)\s+=\s+([\d.]+)"),
        "power_avg_w": grab(r"Power in streaming mode \(average\)\s+=\s+([\d.]+)"),
    }
    LOG.info("[npu] %s", result)
    save_json(out_dir / "npu_benchmark.json", result)
    return result


# =============================================================================================== #
# Lệnh: stress (resource profiler)
# =============================================================================================== #
class ResourceSampler(threading.Thread):
    """Lấy mẫu tài nguyên mỗi `interval` giây cho tiến trình `pid` (và tiến trình con)."""

    def __init__(self, pid: int, interval: float, npu_temp_fn=None, exclude_bytes: int = 0):
        """exclude_bytes: RAM tĩnh không thuộc pipeline suy luận (vd cache ảnh test của stress/python) -
        bị trừ khỏi RSS báo cáo (proc_rss_mb); RSS gốc vẫn được ghi ở proc_rss_gross_mb."""
        super().__init__(daemon=True)
        self.exclude_mb = exclude_bytes / 2**20
        if psutil is None:
            raise SystemExit("[LỖI] Cần psutil: sudo apt install python3-psutil")
        self.proc = psutil.Process(pid)
        self.interval, self.npu_temp_fn = interval, npu_temp_fn
        self.samples: list[dict] = []
        self._halt = threading.Event()  # không dùng tên _stop: trùng method nội bộ của Thread
        self.t0 = time.time()

    def _procs(self):
        """Tiến trình gốc + con. Giữ nguyên đối tượng Process theo PID: cpu_percent() của một đối tượng
        mới luôn trả 0 ở lần gọi đầu, nên phải tái sử dụng để có mốc đo giữa hai lần lấy mẫu."""
        try:
            current = [self.proc] + self.proc.children(recursive=True)
        except psutil.Error:
            return []
        procs = []
        for p in current:
            if p.pid not in self._known:
                self._known[p.pid] = p
                try:
                    p.cpu_percent(None)  # mốc đầu tiên, đóng góp từ lần lấy mẫu sau
                except psutil.Error:
                    continue
            procs.append(self._known[p.pid])
        return procs

    def run(self):
        self._known = {}
        psutil.cpu_percent(None)
        self._procs()
        while not self._halt.wait(self.interval):
            procs = self._procs()
            rss = cpu_proc = 0.0
            for p in procs:
                try:
                    rss += p.memory_info().rss
                    cpu_proc += p.cpu_percent(None)
                except psutil.Error:
                    pass
            vm = psutil.virtual_memory()
            freq = psutil.cpu_freq()
            self.samples.append({
                "t_s": round(time.time() - self.t0, 2),
                "proc_rss_mb": round(rss / 2**20 - self.exclude_mb, 1),  # RAM của pipeline suy luận
                "proc_rss_gross_mb": round(rss / 2**20, 1),
                "proc_cpu_percent": round(cpu_proc, 1),  # có thể > 100 (nhiều lõi)
                "system_cpu_percent": psutil.cpu_percent(None),
                "system_ram_used_mb": round((vm.total - vm.available) / 2**20, 1),
                "cpu_freq_mhz": round(freq.current, 0) if freq else None,
                "soc_temp_c": read_soc_temp(),
                "npu_temp_c": self.npu_temp_fn() if self.npu_temp_fn else None,
            })

    def stop(self):
        self._halt.set()
        self.join(timeout=5)


def summarize_samples(samples: list[dict]) -> dict:
    def col(key):
        return np.array([s[key] for s in samples if s.get(key) is not None], dtype=float)

    out = {}
    for key in ("proc_rss_mb", "proc_rss_gross_mb", "proc_cpu_percent", "system_cpu_percent", "system_ram_used_mb",
                "cpu_freq_mhz", "soc_temp_c", "npu_temp_c"):
        v = col(key)
        if len(v):
            tail = v[-min(len(v), 10):]  # 10 giây cuối = trạng thái nhiệt ổn định
            out[key] = {"mean": float(v.mean()), "max": float(v.max()), "start": float(v[0]),
                        "final": float(tail.mean())}
    return out


def stress_python(args, out_dir: Path) -> dict:
    backend = make_backend(args, args.stress_conf)
    images = sorted(p for p in (args.dataset / "images" / args.split).iterdir() if p.suffix.lower() in IMG_EXTS)
    # Giữ ảnh ở dạng JPEG nén (vài chục KB/ảnh) và giải mã ngay trước mỗi vòng - ngoài phần đo thời gian,
    # giống camera đưa frame vào. Kích thước cache được trừ khỏi RSS để chỉ còn RAM của pipeline suy luận.
    encoded = [np.fromfile(str(p), dtype=np.uint8) for p in images[: args.stress_images]]
    encoded = [e for e in encoded if cv2.imdecode(e, cv2.IMREAD_COLOR) is not None]
    if not encoded:
        backend.close()
        raise SystemExit("[LỖI] Không có ảnh để chạy stress")
    cache_bytes = sum(e.nbytes for e in encoded)
    sampler = ResourceSampler(os.getpid(), args.interval, backend.npu_temperature, exclude_bytes=cache_bytes)
    lat = {"preprocess": [], "inference": [], "postprocess": [], "e2e": []}
    LOG.info("[stress/python] %d s suy luận liên tục trên %d ảnh test (lặp vòng, cache %.1f MB được trừ khỏi RSS)...",
             args.duration, len(encoded), cache_bytes / 2**20)
    sampler.start()
    t_begin = time.perf_counter()
    t_end = t_begin + args.duration
    n = 0
    try:
        while time.perf_counter() < t_end:
            bgr = cv2.imdecode(encoded[n % len(encoded)], cv2.IMREAD_COLOR)  # "camera" - không tính vào độ trễ
            t0 = time.perf_counter()
            rgb, r, pad = preprocess(bgr, backend.size)
            t1 = time.perf_counter()
            raw = backend.infer(rgb)
            t2 = time.perf_counter()
            pred = backend.postprocess(raw)
            scale_boxes_to_original(pred[:, :4], r, pad, bgr.shape[:2])
            t3 = time.perf_counter()
            lat["preprocess"].append((t1 - t0) * 1e3), lat["inference"].append((t2 - t1) * 1e3)
            lat["postprocess"].append((t3 - t2) * 1e3), lat["e2e"].append((t3 - t0) * 1e3)
            n += 1
            if n % 1000 == 0:
                LOG.info("  %d frame | %.1f FPS | SoC %s C", n, n / (time.perf_counter() - t_begin), read_soc_temp())
    finally:
        elapsed = time.perf_counter() - t_begin  # thời lượng THỰC (kể cả khi bị ngắt giữa chừng)
        sampler.stop()
        backend.close()
    return {"engine": "python", "frames": n, "fps": n / elapsed if elapsed > 0 else 0.0,
            "duration_s": round(elapsed, 2), "image_cache_mb": round(cache_bytes / 2**20, 2),
            "latency_ms": {k: {"mean": float(np.mean(v)), "p50": float(np.percentile(v, 50)),
                               "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99))}
                           for k, v in lat.items() if v},
            "samples": sampler.samples}


def stop_process(proc: subprocess.Popen, grace_s: float = 10.0) -> None:
    """SIGTERM để inference_app dừng có trật tự (đóng NPU, ghi nốt CSV); quá hạn thì SIGKILL."""
    if proc.poll() is not None:
        return
    LOG.warning("Dừng inference_app (pid %d)...", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        LOG.warning("inference_app không dừng sau %.0f s - kill", grace_s)
        proc.kill()
        proc.wait()


def summarize_app_npu_temp(rows: list[dict]) -> dict | None:
    """Nhiệt độ NPU do chính app đọc qua HailoRT (cột npu_temp_c, -1 = không đọc được), theo thời gian của app."""
    pts = [(float(r["t_ms"]) / 1e3, float(r["npu_temp_c"])) for r in rows if float(r["npu_temp_c"]) > 0]
    if not pts:
        return None
    t, v = np.array(pts).T
    tail = v[t >= t[-1] - 10.0]  # 10 giây cuối = trạng thái nhiệt ổn định
    return {"mean": float(v.mean()), "max": float(v.max()), "start": float(v[0]), "final": float(tail.mean())}


def stress_cpp(args, out_dir: Path) -> dict:
    app = Path(args.app)
    stats_csv = out_dir / "app_frame_stats.csv"
    stats_csv.unlink(missing_ok=True)
    cmd = [str(app), "--hef", str(args.hef), "--source", args.source, "--headless", "--duration",
           str(args.duration), "--stats", str(stats_csv), "--conf", str(args.stress_conf)]
    if not any(args.source.startswith(p) for p in ("/dev/video", "libcamera")) and not args.source.isdigit():
        cmd.append("--loop")  # file video: phát lặp để đủ thời gian đo
    cmd += shlex.split(args.app_args)
    LOG.info("[stress/cpp] %s", " ".join(cmd))

    rc, sampler, proc = None, None, None
    t_start = time.perf_counter()
    with open(out_dir / "app_stdout.log", "w", encoding="utf-8") as log_file:
        try:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            sampler = ResourceSampler(proc.pid, args.interval)
            sampler.start()
            try:
                rc = proc.wait(timeout=args.duration + 60)
            except subprocess.TimeoutExpired:
                LOG.warning("inference_app chạy quá %d s so với --duration", 60)
        finally:
            # Luôn dọn tiến trình con - kể cả khi script bị Ctrl+C / SIGTERM / lỗi - để không giữ /dev/hailo0
            if sampler is not None:
                sampler.stop()
            if proc is not None:
                stop_process(proc)
                rc = proc.returncode if rc is None else rc
    wall_s = time.perf_counter() - t_start

    rows = list(csv.DictReader(open(stats_csv, encoding="utf-8"))) if stats_csv.is_file() else []
    if not rows:
        raise SystemExit(f"[LỖI] inference_app (mã {rc}) không ghi frame nào - xem {out_dir / 'app_stdout.log'}")
    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    span_s = (col("t_ms")[-1] - col("t_ms")[0]) / 1e3 if len(rows) > 1 else 0.0
    ended_early = rc != 0 or span_s < 0.95 * args.duration
    if ended_early:
        LOG.warning("inference_app dừng sớm: mã thoát %s, chạy thực %.1f s / yêu cầu %d s - số liệu chỉ phản ánh "
                    "%.1f s đầu (xem app_stdout.log)", rc, span_s, args.duration, span_s)
    for s in sampler.samples:
        s["npu_temp_c"] = None  # nhiệt độ NPU lấy từ CSV của app (summarize_app_npu_temp), không nội suy
    keys = {"preprocess": "preprocess_ms", "queue": "queue_ms", "inference": "npu_ms",
            "postprocess": "postprocess_ms", "e2e": "e2e_ms"}
    return {"engine": "cpp", "frames": len(rows), "fps": (len(rows) - 1) / span_s if span_s > 0 else 0.0,
            "duration_s": round(span_s, 2), "wall_s": round(wall_s, 2), "app_exit_code": rc,
            "ended_early": ended_early, "npu_temp_from_app": summarize_app_npu_temp(rows),
            "latency_ms": {k: {"mean": float(col(c).mean()), "p50": float(np.percentile(col(c), 50)),
                               "p95": float(np.percentile(col(c), 95)), "p99": float(np.percentile(col(c), 99))}
                           for k, c in keys.items()},
            "samples": sampler.samples}


def run_stress(args, out_dir: Path) -> dict:
    result = stress_cpp(args, out_dir) if args.engine == "cpp" else stress_python(args, out_dir)
    samples = result.pop("samples")
    if samples:
        with open(out_dir / f"stress_{args.engine}_samples.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(samples[0]))
            w.writeheader()
            w.writerows(samples)
    result["resources"] = summarize_samples(samples)
    if result.get("npu_temp_from_app"):
        result["resources"]["npu_temp_c"] = result.pop("npu_temp_from_app")
    result["throttled"] = read_throttled()
    result["requested_duration_s"] = args.duration
    if result["duration_s"] < 0.95 * args.duration:
        result["ended_early"] = True
        LOG.warning("Chỉ chạy được %.1f s / %d s yêu cầu", result["duration_s"], args.duration)
    LOG.info("[stress/%s] %d frame trong %.1f s, %.1f FPS, e2e %.2f ms | RSS %.0f MB | CPU %.0f%% | SoC %s C | "
             "NPU %s C", args.engine, result["frames"], result["duration_s"], result["fps"],
             result["latency_ms"]["e2e"]["mean"],
             result["resources"].get("proc_rss_mb", {}).get("mean", float("nan")),
             result["resources"].get("system_cpu_percent", {}).get("mean", float("nan")),
             result["resources"].get("soc_temp_c", {}).get("final"),
             result["resources"].get("npu_temp_c", {}).get("final"))
    save_json(out_dir / f"stress_{args.engine}.json", result)
    return result


# =============================================================================================== #
# Bảng tổng hợp theo các nhóm chỉ số của đề bài
# =============================================================================================== #
def build_summary(acc: dict | None, base: dict | None, npu: dict | None, stress: dict | None) -> list[dict]:
    rows = []

    def add(group, metric, value, unit, note=""):
        if value is not None:
            rows.append({"group": group, "metric": metric, "value": round(float(value), 4), "unit": unit,
                         "note": note})

    if acc:
        add("accuracy", "mAP@0.5", acc["mAP50"] * 100, "%", f"{acc['backend']}, {acc['images']} ảnh test")
        add("accuracy", "mAP@0.5:0.95", acc["mAP50_95"] * 100, "%")
        for name, m in acc["per_class"].items():
            add("accuracy", f"mAP@0.5 {name}", m["mAP50"] * 100, "%")
            add("accuracy", f"mAP@0.5:0.95 {name}", m["mAP50_95"] * 100, "%")
        add("accuracy", "Precision", acc["precision"] * 100, "%"), add("accuracy", "Recall", acc["recall"] * 100, "%")
        if base:
            for key, label in (("mAP50", "mAP@0.5"), ("mAP50_95", "mAP@0.5:0.95")):
                diff = acc[key] - base[key]
                add("accuracy", f"Δ{label}", diff * 100, "điểm %",
                    f"NPU - {base['backend']} FP32 (hiệu tuyệt đối; âm = NPU giảm)")
                if base[key] > 0:
                    add("accuracy", f"Δ{label} tương đối", 100 * diff / base[key], "%", "so với baseline FP32")
    if npu:
        add("latency", "NPU inference time (hw)", npu.get("latency_hw_ms"), "ms", "hailortcli benchmark")
        add("latency", "NPU throughput (hw_only)", npu.get("fps_hw_only"), "FPS", "hailortcli benchmark")
    if stress:
        lat = stress["latency_ms"]
        note = f"engine={stress['engine']}, {stress['duration_s']:.0f} s"
        if stress.get("ended_early"):
            note += (f" - DỪNG SỚM (yêu cầu {stress.get('requested_duration_s')} s, "
                     f"mã thoát app {stress.get('app_exit_code', '-')})")
        add("latency", "Preprocessing time", lat.get("preprocess", {}).get("mean"), "ms", note)
        add("latency", "Inference call time", lat.get("inference", {}).get("mean"), "ms", note + " (gồm truyền PCIe)")
        add("latency", "Postprocessing time", lat.get("postprocess", {}).get("mean"), "ms",
            note + " (YOLO26 NMS-free: decode + sigmoid + top-k trên CPU, không NMS)")
        add("latency", "End-to-End latency", lat.get("e2e", {}).get("mean"), "ms", note)
        add("latency", "End-to-End latency p95", lat.get("e2e", {}).get("p95"), "ms", note)
        add("latency", "Throughput", stress["fps"], "FPS", note)
        res = stress["resources"]
        add("resources", "Process RAM (RSS)", res.get("proc_rss_mb", {}).get("mean"), "MB",
            f"đã trừ cache ảnh test {stress['image_cache_mb']} MB" if stress.get("image_cache_mb") else "")
        add("resources", "System RAM used", res.get("system_ram_used_mb", {}).get("mean"), "MB")
        add("resources", "CPU utilization (system)", res.get("system_cpu_percent", {}).get("mean"), "%")
        add("resources", "CPU utilization (process)", res.get("proc_cpu_percent", {}).get("mean"), "%",
            "100% = 1 lõi")
        if npu and npu.get("fps_hw_only"):
            add("resources", "NPU utilization (ước lượng)", 100 * stress["fps"] / npu["fps_hw_only"], "%",
                "FPS thực / FPS hw_only; đo trực tiếp bằng `hailortcli monitor`")
        add("resources", "SoC temperature (sau stress)", res.get("soc_temp_c", {}).get("final"), "°C",
            f"TB 10 s cuối sau {stress['duration_s']:.0f} s chạy thực" + (" (dừng sớm)" if stress.get("ended_early") else ""))
        add("resources", "SoC temperature (max)", res.get("soc_temp_c", {}).get("max"), "°C")
        add("resources", "NPU temperature (sau stress)", res.get("npu_temp_c", {}).get("final"), "°C")
    for name, mb in file_sizes().items():
        add("resources", f"Footprint {name}", mb, "MB")
    return rows


# =============================================================================================== #
# CLI
# =============================================================================================== #
def parse_args():
    p = argparse.ArgumentParser(description="Đo kiểm mAP / độ trễ / tài nguyên trên Pi 5 + Hailo-8",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("command", choices=("accuracy", "npu", "stress", "all"))
    p.add_argument("--hef", type=Path, default=REPO_ROOT / "models" / "best_hailo.hef")
    p.add_argument("--onnx", type=Path, help="best.onnx: backend onnx / baseline ΔmAP trong lệnh all")
    p.add_argument("--backend", choices=("hailo", "onnx"), default="hailo")
    p.add_argument("--dataset", type=Path, default=REPO_ROOT / "datasets" / "processed")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=0, help="Chỉ dùng N ảnh đầu (0 = tất cả)")
    p.add_argument("--conf", type=float, default=0.001, help="Ngưỡng điểm khi đo mAP (chuẩn Ultralytics val)")
    p.add_argument("--max-det", type=int, default=300, help="Top-k của YOLO26 end-to-end (Ultralytics max_det)")
    p.add_argument("--engine", choices=("python", "cpp"), default="python", help="stress: tiến trình được đo")
    p.add_argument("--duration", type=int, default=300, help="stress: số giây chạy liên tục")
    p.add_argument("--interval", type=float, default=1.0, help="stress: chu kỳ lấy mẫu (s)")
    p.add_argument("--stress-conf", type=float, default=0.5, help="stress: ngưỡng tin cậy như khi chạy thật")
    p.add_argument("--stress-images", type=int, default=200,
                   help="stress/python: số ảnh test giữ sẵn (JPEG nén; dung lượng được trừ khỏi RSS)")
    p.add_argument("--app", default=str(REPO_ROOT / "edge_deployment" / "build" / "inference_app"))
    p.add_argument("--source", default=str(REPO_ROOT / "test_video.mp4"), help="stress/cpp: nguồn video")
    p.add_argument("--app-args", default="",
                   help='Tham số thêm cho inference_app, đặt trong ngoặc kép, vd "--width 1920 --height 1080"')
    p.add_argument("--npu-seconds", type=int, default=10, help="npu: thời gian mỗi chế độ đo của hailortcli")
    p.add_argument("--out", type=Path, help="Thư mục kết quả (mặc định edge_deployment/results/<thời gian>)")
    args = p.parse_args()
    args.out = args.out or REPO_ROOT / "edge_deployment" / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.command in ("accuracy", "all") and args.backend == "onnx" and not args.onnx:
        p.error("--backend onnx cần --onnx models/best.onnx")
    return args


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args()
    # SIGTERM (vd hết giờ của hệ thống điều phối) -> SystemExit để mọi khối finally chạy và dọn inference_app
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    setup_logging(args.out)

    # ---- Kiểm tra điều kiện TRƯỚC khi bắt đầu (không để lỗi sau nhiều phút đo) ----
    want_baseline = args.command == "all" and args.onnx and args.backend == "hailo"
    if (want_baseline or args.backend == "onnx") and importlib.util.find_spec("onnxruntime") is None:
        if args.backend == "onnx":
            raise SystemExit("[LỖI] --backend onnx cần onnxruntime (pip install onnxruntime)")
        LOG.warning("Không có onnxruntime - bỏ qua baseline ONNX/ΔmAP (pip install onnxruntime để đo)")
        want_baseline = False
    if args.command in ("stress", "all"):
        if psutil is None:
            raise SystemExit("[LỖI] Cần psutil: sudo apt install python3-psutil")
        if args.engine == "cpp" and not Path(args.app).is_file():
            raise SystemExit(f"[LỖI] Không thấy {args.app}. Build trước: cmake -S edge_deployment -B "
                             f"edge_deployment/build && cmake --build edge_deployment/build -j4")

    info = system_info()
    LOG.info("Thiết bị: %s", info)
    save_json(args.out / "system_info.json", {**info, "args": {k: str(v) for k, v in vars(args).items()}})

    results = {"acc": None, "base": None, "npu": None, "stress": None}
    failures = []

    def stage(key, fn):
        """Mỗi bước độc lập: lỗi ở một bước không làm mất kết quả các bước đã xong."""
        try:
            results[key] = fn()
        except (Exception, SystemExit) as e:  # SystemExit: thông báo [LỖI] có chủ đích từ các bước
            failures.append(f"{key}: {e}")
            LOG.error("Bước %s thất bại: %s", key, e)
            if isinstance(e, SystemExit) and e.code == 143:
                raise

    try:
        if args.command in ("accuracy", "all"):
            stage("acc", lambda: run_accuracy(args, args.out))
            if want_baseline:
                stage("base", lambda: run_accuracy(args, args.out, backend_name="onnx"))
        if args.command in ("npu", "all") and args.backend == "hailo":
            stage("npu", lambda: run_npu_benchmark(args, args.out))
        if args.command in ("stress", "all"):
            stage("stress", lambda: run_stress(args, args.out))
    finally:
        write_summary(args, info, results, failures)
    if failures:
        raise SystemExit(f"[LỖI] {len(failures)} bước thất bại - xem {args.out / 'benchmark.log'}")


def write_summary(args, info, results, failures):
    acc, base, npu, stress = results["acc"], results["base"], results["npu"], results["stress"]
    rows = build_summary(acc, base, npu, stress)
    if rows:
        with open(args.out / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["group", "metric", "value", "unit", "note"])
            w.writeheader()
            w.writerows(rows)
        save_json(args.out / "summary.json", {"system": info, "rows": rows, "accuracy": acc, "baseline": base,
                                              "npu": npu, "stress": stress, "failures": failures})
        LOG.info("\n%-11s %-34s %12s  %s", "group", "metric", "value", "unit")
        for r in rows:
            LOG.info("%-11s %-34s %12.3f  %s  %s", r["group"], r["metric"], r["value"], r["unit"], r["note"])
    LOG.info("Kết quả: %s", args.out)


if __name__ == "__main__":
    main()
