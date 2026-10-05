#!/usr/bin/env python3
"""
calibrate_ptq.py - Giai đoạn 3.2: Tạo Calibration Dataset cho PTQ INT8 trên Hailo DFC.

Chọn N ảnh tiêu biểu (mặc định 200) từ tập train, tiền xử lý GIỐNG HỆT lúc suy luận rồi lưu thành
một file .npy dạng NHWC mà compile_npu.py nạp vào `ClientRunner.optimize()` (cũng dùng được với CLI
`hailo optimize --calib-set-path`).

Chọn ảnh "tiêu biểu" (tái lập được qua --seed):
  - Mỗi ảnh gốc chỉ lấy 1 bản - bản "sạch" nhất trong các bản augment Roboflow ".rf.<hash>".
  - Phân tầng theo nội dung: ảnh có pedestrian / ảnh chỉ có traffic_sign / ảnh nền, theo tỷ lệ của
    tập train nhưng pedestrian chiếm ít nhất --min-ped-share (lớp hiếm vẫn có mặt trong dải kích hoạt).
  - Trong mỗi tầng, chia số ảnh cho các chuỗi video (họ tên file) theo căn bậc hai kích thước họ
    (chuỗi lớn được nhiều ảnh hơn nhưng chuỗi nhỏ không bị bỏ quên), và trong mỗi chuỗi lấy các frame
    trải đều theo thời gian -> phủ nhiều cảnh/ánh sáng, tránh frame liền kề gần trùng nhau.
  - Bỏ các bản bị Roboflow augment mạnh (xoay có góc đen, ảnh xám, nhiễu muối tiêu): chúng không giống
    ảnh camera thật trên xe và làm dải kích hoạt INT8 bị nới rộng vô ích.

Tiền xử lý (khớp ultralytics LetterBox và chương trình suy luận trên Pi):
  - Letterbox về --imgsz x --imgsz, giữ tỷ lệ khung hình, viền màu (114, 114, 114), nội suy INTER_LINEAR.
  - BGR -> RGB, layout NHWC.
  - Chuẩn hóa pixel: mặc định --normalize npu -> GIỮ dải 0..255 (uint8). Phép chia 255 được đưa vào NPU
    bằng lớp `normalization([0,0,0], [255,255,255])` trong model script của compile_npu.py (chuẩn Hailo
    Model Zoo): Pi chỉ việc đẩy ảnh uint8 vào NPU, không tốn CPU chuẩn hóa. --normalize host -> lưu
    float32 0..1 và compile_npu.py sẽ không thêm lớp normalization.

Đầu ra:
  models/calib/calib_set_640.npy       (N, 640, 640, 3) uint8 RGB
  models/calib/calib_set_640.json      metadata (tiền xử lý, danh sách ảnh, thống kê mean/std)
  models/calib/calib_set_640_preview.jpg   lưới ảnh minh họa cho báo cáo

Ví dụ:
  python optimization/calibrate_ptq.py
  python optimization/calibrate_ptq.py --num 1024 --seed 0      # cho optimization_level=2 trên server GPU
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)
HEX_HASH = re.compile(r"[0-9a-f]{16,}", re.IGNORECASE)
DIGITS = re.compile(r"\d+")
PAD_VALUE = 114
GOLDEN = 0.6180339887498949


# --------------------------------------------------------------------------- #
# Tiền xử lý
# --------------------------------------------------------------------------- #
def imread(path):
    """cv2.imread không mở được đường dẫn có ký tự Unicode (vd 'Tài liệu') trên Windows."""
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def letterbox(img, size, pad=PAD_VALUE):
    """Resize giữ tỷ lệ + căn giữa, khớp ultralytics.data.augment.LetterBox(auto=False, center=True)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = round(w * r), round(h * r)
    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad, pad, pad))


def preprocess(path, size):
    img = imread(path)
    if img is None:
        raise ValueError(f"không đọc được ảnh {path}")
    img = letterbox(img, size)
    return np.ascontiguousarray(img[:, :, ::-1])  # BGR -> RGB, HWC uint8


# --------------------------------------------------------------------------- #
# Chọn ảnh tiêu biểu
# --------------------------------------------------------------------------- #
def describe(img_path, labels_dir):
    """Trả về (tầng nội dung, họ chuỗi video, ảnh gốc, thứ tự frame) của một ảnh."""
    lbl = labels_dir / (img_path.stem + ".txt")
    classes = set()
    if lbl.is_file():
        classes = {line.split()[0] for line in lbl.read_text(encoding="utf-8").splitlines() if line.strip()}
    stratum = "pedestrian" if "1" in classes else "traffic_sign" if classes else "background"
    source = ROBOFLOW_SUFFIX.sub("", img_path.stem)
    clean = HEX_HASH.sub("", source)
    return stratum, DIGITS.sub("#", clean), source, tuple(int(n) for n in DIGITS.findall(clean))


def augmentation_flags(path):
    """Dấu hiệu ảnh bị Roboflow augment mạnh. Ngưỡng hiệu chỉnh trên dữ liệu thật: 0% dương tính giả
    trên các chuỗi chưa từng bị augment."""
    img = imread(path)
    if img is None:
        return {"unreadable"}
    flags = set()
    bright = img.mean() > 40  # bỏ qua cảnh tối thật để không bắt nhầm
    h, w = img.shape[:2]
    k = max(4, int(min(h, w) * 0.06))
    corners = (img[:k, :k], img[:k, -k:], img[-k:, :k], img[-k:, -k:])
    if bright and max((c.reshape(-1, 3).max(1) < 6).mean() for c in corners) > 0.5:
        flags.add("rotated")  # nền đen tuyệt đối do xoay/shear
    b, g, r = (img[..., i].astype(np.int16) for i in range(3))
    if bright and (np.abs(r - g) + np.abs(g - b)).mean() < 3:
        flags.add("grayscale")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    isolated = (np.abs(gray.astype(np.int16) - cv2.medianBlur(gray, 3)) > 60) & ((gray < 25) | (gray > 230))
    if isolated.mean() > 0.002:
        flags.add("noise")  # điểm cực trị cô lập = nhiễu muối tiêu
    return flags


def spread_order(n, offset):
    """Thứ tự duyệt 0..n-1 mà k phần tử đầu luôn trải đều trên cả chuỗi (dãy tỷ lệ vàng)."""
    order, seen = [], set()
    for k in range(4 * n):
        i = int(((offset + k * GOLDEN) % 1.0) * n)
        if i not in seen:
            seen.add(i)
            order.append(i)
    return order + [i for i in range(n) if i not in seen]


def allocate(total, sizes):
    """Chia `total` cho các nhóm theo căn bậc hai kích thước, không vượt kích thước mỗi nhóm."""
    quota = {k: 0 for k in sizes}
    active = sorted(k for k, v in sizes.items() if v > 0)
    while sum(quota.values()) < total and active:
        remaining = total - sum(quota.values())
        wsum = sum(math.sqrt(sizes[k]) for k in active)
        raw = {k: remaining * math.sqrt(sizes[k]) / wsum for k in active}
        give = {k: min(sizes[k] - quota[k], int(raw[k])) for k in active}
        if not any(give.values()):  # phần dư làm tròn -> phát cho phần lẻ lớn nhất
            for k in sorted(active, key=lambda k: (int(raw[k]) - raw[k], k))[:remaining]:
                give[k] = 1
        for k, v in give.items():
            quota[k] += v
        active = [k for k in active if quota[k] < sizes[k]]
    return quota


def stratum_quotas(pool, num, min_ped_share):
    total = sum(len(v) for v in pool.values())
    share = {k: len(v) / total for k, v in pool.items()}
    if pool.get("pedestrian"):
        extra = max(0.0, min_ped_share - share["pedestrian"])
        rest = sum(s for k, s in share.items() if k != "pedestrian")
        for k in share:
            share[k] += extra if k == "pedestrian" else -extra * share[k] / rest if rest else 0
    quota = {k: min(len(pool[k]), round(num * s)) for k, s in share.items()}
    for k in sorted(quota, key=lambda k: -len(pool[k])):  # bù/trừ sai số làm tròn
        diff = num - sum(quota.values())
        if diff == 0:
            break
        quota[k] = max(0, min(len(pool[k]), quota[k] + diff))
    return quota


def pick_stratum(entries, quota, rng, ex, skip_augmented):
    """Chọn `quota` ảnh gốc trong một tầng: phân bổ theo họ chuỗi, frame trải đều, bỏ ảnh augment."""
    families = defaultdict(list)
    for e in entries:
        families[e["family"]].append(e)
    orders = {}
    for fam in sorted(families):
        seq = sorted(families[fam], key=lambda e: (e["order"], e["source"]))
        orders[fam] = [seq[i] for i in spread_order(len(seq), rng.random())]
    need = allocate(quota, {f: len(v) for f, v in orders.items()})
    cursor = {f: 0 for f in orders}
    picked, reserve, unreadable = [], [], 0

    def take(fam, k):
        cands = orders[fam][cursor[fam]:cursor[fam] + k]
        cursor[fam] += len(cands)
        return cands

    def best_copy(entry):
        """Bản ít dấu hiệu augment nhất trong các bản Roboflow của một ảnh gốc."""
        scored = [(len(f), i, p, f) for i, p in enumerate(entry["copies"]) for f in [augmentation_flags(p)]]
        _, _, path, flags = min(scored, key=lambda s: ("unreadable" in s[3], s[0], s[1]))
        return {**entry, "path": path, "flags": sorted(flags)}

    def consume(cands, per_family_need):
        nonlocal unreadable
        for cand in ex.map(best_copy, cands):
            if len(picked) >= quota:
                break
            if "unreadable" in cand["flags"]:
                unreadable += 1
            elif skip_augmented and cand["flags"]:
                reserve.append(cand)
            else:
                picked.append(cand)
                if per_family_need:
                    need[cand["family"]] -= 1

    while len(picked) < quota:  # 1) theo quota từng họ chuỗi
        batch = [c for f in sorted(need) if need[f] > 0 for c in take(f, need[f])]
        if not batch:
            break
        consume(batch, True)
    while len(picked) < quota:  # 2) họ nào cạn ảnh sạch -> bù xoay vòng từ các họ còn ứng viên
        batch = [c for f in sorted(orders) for c in take(f, 1)]
        if not batch:
            break
        consume(batch, False)
    if len(picked) < quota:  # 3) vẫn thiếu -> dùng ảnh ít dấu hiệu augment nhất
        reserve.sort(key=lambda c: (len(c["flags"]), c["source"]))
        picked += reserve[:quota - len(picked)]
    return picked, reserve, unreadable


def select_images(images_dir, num, min_ped_share, seed, skip_augmented, workers):
    labels_dir = images_dir.parent.parent / "labels" / images_dir.name
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not paths:
        sys.exit(f"[LỖI] Không có ảnh nào trong {images_dir}")
    if not labels_dir.is_dir():
        print(f"[CẢNH BÁO] Không thấy {labels_dir} - không phân tầng được theo lớp.")

    sources = {}  # gom các bản augment Roboflow của cùng một ảnh gốc
    for p in paths:
        stratum, family, source, order = describe(p, labels_dir)
        entry = sources.setdefault(source, {"source": source, "stratum": stratum, "family": family,
                                            "order": order, "copies": []})
        entry["copies"].append(p)
    pool = defaultdict(list)
    for source in sorted(sources):
        pool[sources[source]["stratum"]].append(sources[source])

    num = min(num, len(sources))
    quota = stratum_quotas(pool, num, min_ped_share)
    rng = random.Random(seed)
    chosen, rejected, unreadable = [], Counter(), 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for stratum in sorted(pool):
            picked, reserve, bad = pick_stratum(pool[stratum], quota[stratum], rng, ex, skip_augmented)
            chosen += picked
            unreadable += bad
            rejected.update(f for c in reserve for f in c["flags"])
    if unreadable > max(5, 0.05 * num):
        sys.exit(f"[LỖI] {unreadable} ảnh không đọc được - kiểm tra lại thư mục {images_dir}.")
    rng.shuffle(chosen)
    info = {"unique_sources": len(sources), "pool": {k: len(v) for k, v in pool.items()},
            "rejected_augmented_flags": dict(rejected), "unreadable": unreadable,
            "augmented_used": sum(bool(c["flags"]) for c in chosen)}
    return chosen, info


# --------------------------------------------------------------------------- #
# Ghi kết quả
# --------------------------------------------------------------------------- #
def save_preview(array, path, cols=8, thumb=160):
    n = min(len(array), cols * 4)
    rows = (n + cols - 1) // cols
    grid = np.full((rows * thumb, cols * thumb, 3), 255, dtype=np.uint8)
    for i in range(n):
        tile = array[i]
        if tile.dtype != np.uint8:
            tile = np.clip(tile * 255.0, 0, 255).astype(np.uint8)
        r, c = divmod(i, cols)
        grid[r * thumb:(r + 1) * thumb, c * thumb:(c + 1) * thumb] = cv2.resize(tile, (thumb, thumb))
    ok, buf = cv2.imencode(".jpg", grid[:, :, ::-1])
    if ok:
        buf.tofile(str(path))


def channel_stats(data, imgsz):
    s1, s2, lo, hi = np.zeros(3), np.zeros(3), np.inf, -np.inf
    for start in range(0, len(data), 16):  # cộng dồn theo khối để không tốn vài GB RAM
        block = np.asarray(data[start:start + 16], dtype=np.float64).reshape(-1, 3)
        s1 += block.sum(0)
        s2 += (block ** 2).sum(0)
        lo, hi = min(lo, block.min()), max(hi, block.max())
    count = len(data) * imgsz * imgsz
    mean = s1 / count
    std = np.sqrt(np.maximum(s2 / count - mean ** 2, 0))
    return mean.round(3).tolist(), std.round(3).tolist(), float(lo), float(hi)


def parse_args():
    p = argparse.ArgumentParser(description="Tạo Calibration Dataset (.npy) cho Hailo DFC.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--images", type=Path, default=REPO_ROOT / "datasets" / "processed" / "images" / "train")
    p.add_argument("--num", type=int, default=200, help="Số ảnh calibration (khuyến nghị 100-500; 1024 cho opt-level 2)")
    p.add_argument("--imgsz", type=int, default=640, help="Kích thước đầu vào của mô hình (bội số của 32)")
    p.add_argument("--normalize", choices=("npu", "host"), default="npu",
                   help="npu = lưu uint8 0..255, chia 255 trong NPU; host = lưu float32 0..1")
    p.add_argument("--min-ped-share", type=float, default=0.25,
                   help="Tỷ lệ tối thiểu ảnh có pedestrian trong tập calibration")
    p.add_argument("--keep-augmented", action="store_true",
                   help="Không lọc ảnh bị Roboflow augment mạnh (xoay, ảnh xám, nhiễu muối tiêu)")
    p.add_argument("--out", type=Path, help="Mặc định: models/calib/calib_set_<imgsz>.npy")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.imgsz % 32:
        p.error("--imgsz phải là bội số của 32")
    if args.num < 1 or not 0 <= args.min_ped_share < 1 or args.workers < 1:
        p.error("--num, --workers >= 1 và 0 <= --min-ped-share < 1")
    args.out = (args.out or REPO_ROOT / "models" / "calib" / f"calib_set_{args.imgsz}.npy").resolve()
    if args.out.suffix != ".npy":
        p.error("--out phải có đuôi .npy")
    return args


def main():
    for stream in (sys.stdout, sys.stderr):  # tránh UnicodeEncodeError trên console Windows
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args()
    images_dir = args.images.resolve()
    if not images_dir.is_dir():
        sys.exit(f"[LỖI] Không tìm thấy {images_dir}. Chạy data_engineering/clean_and_filter.py trước.")

    print(f"[1/4] Chọn {args.num} ảnh tiêu biểu từ {images_dir} ...")
    chosen, pool_info = select_images(images_dir, args.num, args.min_ped_share, args.seed,
                                      not args.keep_augmented, args.workers)
    strata = Counter(c["stratum"] for c in chosen)
    families = Counter(c["family"] for c in chosen)
    print(f"      {pool_info['unique_sources']} ảnh gốc khả dụng {pool_info['pool']}")
    print(f"      đã chọn {len(chosen)} ảnh: {dict(strata)} | phủ {len(families)} chuỗi video "
          f"(nhiều nhất {max(families.values())} ảnh/chuỗi)")
    print(f"      loại ứng viên bị augment mạnh (Roboflow): {pool_info['rejected_augmented_flags']} | "
          f"buộc phải dùng {pool_info['augmented_used']} ảnh augment | không đọc được: {pool_info['unreadable']}")
    if len(chosen) < 100:
        print("[CẢNH BÁO] Dưới 100 ảnh calibration - dải kích hoạt INT8 có thể kém đại diện.")

    print(f"[2/4] Tiền xử lý: letterbox {args.imgsz}x{args.imgsz} (pad {PAD_VALUE}), BGR->RGB, NHWC, "
          f"{'uint8 0..255 (chuẩn hóa trong NPU)' if args.normalize == 'npu' else 'float32 0..1'} ...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Ghi vào file tạm rồi đổi tên: lần chạy lỗi giữa chừng không để lại .npy chứa khung hình đen
    tmp_npy = args.out.with_name(args.out.stem + ".partial.npy")
    meta_path = args.out.with_suffix(".json")
    dtype = np.uint8 if args.normalize == "npu" else np.float32
    shape = (len(chosen), args.imgsz, args.imgsz, 3)
    data = np.lib.format.open_memmap(tmp_npy, mode="w+", dtype=dtype, shape=shape)

    def load(i):
        x = preprocess(chosen[i]["path"], args.imgsz)
        data[i] = x if dtype == np.uint8 else x.astype(np.float32) / 255.0

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for done, _ in enumerate(ex.map(load, range(len(chosen))), 1):
                if done % 50 == 0 or done == len(chosen):
                    print(f"      {done}/{len(chosen)} ảnh")
        data.flush()
    except BaseException:
        data._mmap.close()
        tmp_npy.unlink(missing_ok=True)
        raise

    print("[3/4] Thống kê kênh màu ...")
    mean, std, lo, hi = channel_stats(data, args.imgsz)
    print(f"      mean RGB = {mean} | std RGB = {std} | min = {lo:.3f} | max = {hi:.3f}")
    preview = args.out.with_name(args.out.stem + "_preview.jpg")
    save_preview(data, preview)
    data._mmap.close()  # Windows: phải đóng memmap trước khi đổi tên file

    print("[4/4] Ghi file ...")
    meta_path.unlink(missing_ok=True)  # metadata cũ không còn khớp
    os.replace(tmp_npy, args.out)
    meta = {
        "file": args.out.name,
        "shape": list(shape),
        "dtype": np.dtype(dtype).name,
        "layout": "NHWC",
        "color": "RGB",
        "imgsz": args.imgsz,
        "letterbox_pad": PAD_VALUE,
        "normalization_in_npu": args.normalize == "npu",
        "pixel_range": [0, 255] if args.normalize == "npu" else [0, 1],
        "channel_mean": mean,
        "channel_std": std,
        "seed": args.seed,
        "strata": dict(strata),
        "families": dict(families),
        "augmentation_filter": {"enabled": not args.keep_augmented,
                                "rejected_flags": pool_info["rejected_augmented_flags"],
                                "augmented_used": pool_info["augmented_used"]},
        "images": [c["path"].name for c in chosen],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nHoàn tất! Calibration set ({args.out.stat().st_size / 2**20:.1f} MB):")
    print(f"  - {args.out}\n  - {meta_path}\n  - {preview}")


if __name__ == "__main__":
    main()
