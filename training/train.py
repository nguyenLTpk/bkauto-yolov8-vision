#!/usr/bin/env python3
"""
train.py - Giai đoạn 2: Huấn luyện Baseline YOLOv8n trên server RTX 5070 Ti.

Siêu tham số đọc từ training/configs/yolov8n_baseline.yaml (có thể ghi đè bằng CLI).

Về image_weights:
  Ultralytics YOLOv8 (>= 8.1, kể cả 8.4.x) ĐÃ GỠ tham số `image_weights` của YOLOv5 - truyền vào
  model.train() sẽ báo lỗi "not a valid YOLO argument". Script này cài lại tính năng đó bằng một
  DetectionTrainer tùy biến: dataloader train dùng WeightedRandomSampler, trọng số mỗi ảnh theo
  Repeat Factor Sampling (Gupta et al., LVIS, CVPR 2019):
      r(c) = (N_max / N_c) ** power       (N_c = số box của lớp c trong tập train)
      w(I) = max_{c in I} r(c)            (ảnh nền: w = 1)
  Với power = 0.5 và tỷ lệ traffic_sign : pedestrian ~ 8.9 : 1, ảnh có người đi bộ được lấy mẫu
  thường hơn ~3 lần - tránh lặp quá nhiều (overfit) như cân bằng tuyệt đối (power = 1).

Early Stopping / best.pt: Ultralytics 8.4 dùng fitness = mAP@0.5:0.95 trên tập val.

Ví dụ:
  python training/train.py
  python training/train.py --epochs 100 --batch 64 --name yolov8n_bs64
  python training/train.py --resume runs/train/yolov8n_baseline/weights/last.pt
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO, settings
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "training" / "configs" / "yolov8n_baseline.yaml"
CUSTOM_KEYS = ("tensorboard", "image_weights", "image_weights_power", "best_weights_out")


# --------------------------------------------------------------------------- #
# Lấy mẫu ảnh có trọng số (thay cho image_weights của YOLOv5)
# --------------------------------------------------------------------------- #
def compute_image_weights(labels, nc, power):
    """Trả về (trọng số từng ảnh, ma trận số box [ảnh x lớp], hệ số lặp mỗi lớp)."""
    per_image = np.stack([np.bincount(lb["cls"].reshape(-1).astype(int), minlength=nc)[:nc] for lb in labels])
    class_counts = per_image.sum(0)
    repeat = (class_counts.max() / np.maximum(class_counts, 1)) ** power
    repeat[class_counts == 0] = 1.0
    weights = np.where(per_image > 0, repeat[None, :], 0.0).max(1)
    weights[weights == 0] = 1.0  # ảnh nền giữ trọng số 1
    return weights, per_image, repeat


class BalancedDetectionTrainer(DetectionTrainer):
    """DetectionTrainer có dataloader train lấy mẫu theo trọng số ảnh."""

    image_weights_power = 0.5

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        if mode != "train" or self.image_weights_power <= 0:
            return loader
        if rank != -1:
            print("[CẢNH BÁO] image_weights không hỗ trợ DDP nhiều GPU - dùng lấy mẫu ngẫu nhiên đều.")
            return loader

        dataset = loader.dataset
        names = self.data["names"]
        weights, per_image, repeat = compute_image_weights(dataset.labels, len(names), self.image_weights_power)
        generator = torch.Generator().manual_seed(self.args.seed)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(dataset),
                                        replacement=True, generator=generator)

        counts = per_image.sum(0)
        expected = (weights / weights.sum()) @ per_image * len(dataset)  # số box kỳ vọng mỗi epoch
        LOGGER.info(f"image_weights (power={self.image_weights_power}):")
        for c, name in names.items():
            LOGGER.info(f"  {name:<14} boxes={counts[c]:>6}  repeat={repeat[c]:.2f}  "
                        f"boxes/epoch {counts[c]:>6} -> {expected[c]:>8.0f}")

        loader.close()
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            collate_fn=loader.collate_fn,
            worker_init_fn=seed_worker,
            generator=loader.generator,
            drop_last=loader.drop_last,
            prefetch_factor=loader.prefetch_factor,
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Huấn luyện Baseline YOLOv8n (BKAuto).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="File YAML siêu tham số")
    p.add_argument("--data", type=str, help="Ghi đè đường dẫn data.yaml")
    p.add_argument("--model", type=str, help="Ghi đè trọng số khởi tạo (vd yolov8s.pt)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=float, help="Số nguyên, hoặc 0-1 = tỷ lệ VRAM cho AutoBatch")
    p.add_argument("--imgsz", type=int)
    p.add_argument("--device", type=str, help="vd 0, cpu")
    p.add_argument("--workers", type=int)
    p.add_argument("--name", type=str, help="Tên thư mục chạy trong runs/train/")
    p.add_argument("--image-weights-power", type=float, help="0 = tắt image_weights")
    p.add_argument("--exist-ok", action="store_true", help="Ghi đè thư mục chạy trùng tên")
    p.add_argument("--resume", type=Path, metavar="LAST_PT",
                   help="Tiếp tục lần chạy bị ngắt từ runs/train/<name>/weights/last.pt")
    return p.parse_args()


def load_config(args):
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    # Đường dẫn tương đối trong file YAML được hiểu theo gốc repo
    for key in ("data", "project", "best_weights_out"):
        if not Path(cfg[key]).is_absolute():
            cfg[key] = str(REPO_ROOT / cfg[key])
    # Đường dẫn gõ trên CLI được hiểu theo thư mục hiện hành
    overrides = {"data": args.data and str(Path(args.data).resolve()),
                 "model": args.model and (str(Path(args.model).resolve()) if Path(args.model).exists() else args.model),
                 "epochs": args.epochs, "imgsz": args.imgsz, "device": args.device, "workers": args.workers,
                 "name": args.name, "image_weights_power": args.image_weights_power}
    if args.batch is not None:
        overrides["batch"] = int(args.batch) if args.batch >= 1 else args.batch
    if args.exist_ok:
        overrides["exist_ok"] = True
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    if cfg["imgsz"] % 32:
        sys.exit(f"[LỖI] imgsz={cfg['imgsz']} phải là bội số của 32 (yêu cầu của NPU).")
    if not args.resume and not Path(cfg["data"]).is_file():
        sys.exit(f"[LỖI] Không tìm thấy {cfg['data']}.\n"
                 f"       Chạy trước: python data_engineering/clean_and_filter.py --src datasets/raw --dst datasets/processed")
    return cfg


def main():
    for stream in (sys.stdout, sys.stderr):  # tránh UnicodeEncodeError trên console Windows
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args()
    cfg = load_config(args)
    custom = {k: cfg.pop(k) for k in CUSTOM_KEYS if k in cfg}

    # Lưu vào settings.json toàn cục của Ultralytics - ghi cả hai chiều để cờ trong config luôn có hiệu lực
    settings.update({"tensorboard": bool(custom.get("tensorboard", True))})
    power = custom.get("image_weights_power", 0.5) if custom.get("image_weights", True) else 0.0
    BalancedDetectionTrainer.image_weights_power = power
    if power > 0 and "," in str(cfg.get("device", "")):
        sys.exit("[LỖI] image_weights chỉ hỗ trợ 1 GPU (DDP dùng DistributedSampler). "
                 "Chọn --device 0 hoặc tắt bằng --image-weights-power 0.")

    if torch.cuda.is_available():
        LOGGER.info(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda}")

    try:
        if args.resume:
            # Ultralytics đọc lại toàn bộ tham số từ checkpoint; trainer tùy biến giữ lại image_weights
            model = YOLO(str(args.resume.resolve()))
            model.train(trainer=BalancedDetectionTrainer, resume=True)
        else:
            model = YOLO(cfg.pop("model"))
            model.train(trainer=BalancedDetectionTrainer, **cfg)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # trả VRAM cho người dùng chung GPU

    trainer = model.trainer
    best = Path(trainer.best)
    if not best.is_file():
        sys.exit(f"[LỖI] Không tìm thấy {best} - quá trình huấn luyện có thể đã bị ngắt.")
    out = Path(custom.get("best_weights_out", REPO_ROOT / "models" / "best.pt"))
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)

    # Dùng print thay LOGGER: LOGGER của Ultralytics xóa ký tự non-ASCII (tiếng Việt) trên Windows
    print(f"\nNhật ký huấn luyện : {trainer.save_dir}  (tensorboard --logdir {trainer.save_dir.parent})")
    print(f"Mô hình tốt nhất   : {best}  ->  {out}  ({out.stat().st_size / 2**20:.2f} MB)")
    m = trainer.metrics or getattr(model.metrics, "results_dict", None) or {}
    print(f"Val (best.pt)      : mAP@0.5 = {m.get('metrics/mAP50(B)', float('nan')):.4f} | "
          f"mAP@0.5:0.95 = {m.get('metrics/mAP50-95(B)', float('nan')):.4f}")


if __name__ == "__main__":
    main()
