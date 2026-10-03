#!/usr/bin/env python3
"""
clean_and_filter.py - Giai đoạn 1: Lọc lớp, làm sạch nhãn, xử lý ảnh nền và chia tập dữ liệu.

Pipeline (chạy tuần tự, kết quả tái lập được nhờ --seed):
  1. Dò tìm toàn bộ cặp ảnh/nhãn YOLO trong --src (gộp mọi split gốc train/valid/test,
     hỗ trợ cả layout images/<split> + labels/<split> lẫn <split>/images + <split>/labels).
  2. Lọc lớp & đánh lại chỉ mục:
        class gốc 0..8  -> 0 (traffic_sign)
        class gốc 9     -> 1 (pedestrian)
        class gốc 10..16 -> xóa bounding box
  3. Làm sạch:
        - bỏ ảnh 0 bytes / ảnh hỏng (không giải mã được bằng PIL);
        - bỏ box lỗi: tâm (x, y) ngoài [0, 1], w/h <= 0, NaN/Inf, dòng sai định dạng;
        - box tràn biên một phần được cắt (clip) về [0, 1];
        - bỏ box quá nhỏ (< --min-box-px) hoặc tỷ lệ khung hình bất thường (> --max-aspect);
        - bỏ box trùng lặp; dòng polygon (segment) được quy đổi về bbox bao ngoài.
  4. Ảnh nền (background negatives): giữ ngẫu nhiên số ảnh không còn nhãn sao cho chúng
     chiếm --bg-ratio (mặc định 10%) tổng số ảnh đầu ra; phần dư bị loại.
  5. Chia Train/Val/Test (mặc định 80:10:10) theo nhóm + phân tầng:
        - các bản augment Roboflow của cùng một ảnh gốc (".rf.<hash>") luôn chung một nhóm;
        - các frame liên tiếp của cùng một chuỗi video được gom theo cụm --group-size frame
          để tránh rò rỉ dữ liệu (data leakage) giữa Train và Test;
        - nhóm được phân bổ tham lam sao cho số box traffic_sign, số box pedestrian, số ảnh
          và số ảnh nền của mỗi tập bám sát tỷ lệ mục tiêu (stratified group split).
  6. Ghi dữ liệu đầu ra theo chuẩn Ultralytics + data.yaml, manifest CSV và báo cáo JSON.

Ví dụ:
  python data_engineering/clean_and_filter.py --src datasets/raw --dst datasets/processed
"""

import argparse
import bisect
import csv
import json
import math
import os
import random
import re
import shutil
import stat
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # tqdm là tùy chọn
    tqdm = None

# --------------------------------------------------------------------------- #
# Cấu hình lớp
# --------------------------------------------------------------------------- #
CLASS_MAP = {**{c: 0 for c in range(9)}, 9: 1}  # class gốc -> class mới
DROP_CLASSES = set(range(10, 17))  # Car, đèn giao thông, Closed road stand, Parking spot, Stop line
NEW_NAMES = {0: "traffic_sign", 1: "pedestrian"}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
OUT_SPLITS = ("train", "val", "test")
ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)
HEX_HASH = re.compile(r"[0-9a-f]{16,}", re.IGNORECASE)
DIGITS = re.compile(r"\d+")


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #
def progress(iterable, total, desc):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit="img")
    return _simple_progress(iterable, total, desc)


def _simple_progress(iterable, total, desc):
    step = max(1, total // 20)
    for i, item in enumerate(iterable, 1):
        if i % step == 0 or i == total:
            print(f"  {desc}: {i}/{total} ({100 * i / total:.0f}%)", flush=True)
        yield item


def is_relative_to(path, other):
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def read_source_names(src):
    """Đọc tên lớp gốc từ data.yaml đầu tiên tìm thấy (chỉ để in bảng ánh xạ)."""
    for yaml_path in sorted(src.rglob("data.yaml")):
        names, in_names = {}, False
        for line in yaml_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if re.match(r"^names\s*:", line):
                in_names = True
                inline = line.split(":", 1)[1].strip()
                if inline.startswith("["):  # names: [a, b, c]
                    items = [s.strip().strip("'\"") for s in inline.strip("[]").split(",")]
                    return dict(enumerate(items))
                continue
            if in_names:
                m = re.match(r"^\s+(\d+)\s*:\s*(.+)$", line) or re.match(r"^\s+-\s*(.+)$", line)
                if m is None:
                    break
                if m.lastindex == 2:
                    names[int(m.group(1))] = m.group(2).strip().strip("'\"")
                else:
                    names[len(names)] = m.group(1).strip().strip("'\"")
        if names:
            return names
    return {}


# --------------------------------------------------------------------------- #
# Bước 1: dò tìm cặp ảnh / nhãn
# --------------------------------------------------------------------------- #
def find_label(src, rel):
    """Quy ước Ultralytics: thay thư mục 'images' gần nhất (bên trong --src) bằng 'labels', đuôi -> .txt.
    Không có thì thử file .txt nằm cạnh ảnh."""
    parts = list(rel.parts)
    candidates = []
    for i in range(len(parts) - 2, -1, -1):
        if parts[i].lower() == "images":
            for labels in ("labels", "Labels", "LABELS"):
                candidates.append(src.joinpath(*parts[:i], labels, *parts[i + 1:]).with_suffix(".txt"))
            break
    candidates.append((src / rel).with_suffix(".txt"))
    return next((c for c in candidates if c.is_file()), None)


def discover_samples(src, exclude):
    """Trả về (danh sách mẫu ảnh, số file nhãn mồ côi không có ảnh tương ứng)."""
    samples, label_files = [], set()
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        dirs[:] = sorted(d for d in dirs if not (exclude and (root_path / d).resolve() == exclude))
        for name in files:
            suffix = Path(name).suffix.lower()
            if suffix == ".txt" and "labels" in (p.lower() for p in root_path.relative_to(src).parts):
                label_files.add(root_path / name)
            if suffix in IMG_EXTS:
                rel = (root_path / name).relative_to(src)
                # split gốc = thư mục train/valid/test gần nhất (để truy vết, không dùng để chia)
                origin = next((p for p in reversed(rel.parts[:-1]) if p.lower() in ("train", "valid", "val", "test")),
                              "-")
                samples.append({"img": src / rel, "lbl": find_label(src, rel), "rel": rel.as_posix(),
                                "origin": origin})
    samples.sort(key=lambda s: s["rel"])  # thứ tự cố định giữa các hệ điều hành
    # file .txt trong thư mục labels/ mà không ảnh nào dùng tới
    orphans = len(label_files - {s["lbl"] for s in samples if s["lbl"] is not None})
    return samples, orphans


# --------------------------------------------------------------------------- #
# Bước 2 + 3: kiểm tra ảnh, lọc lớp, làm sạch nhãn
# --------------------------------------------------------------------------- #
def check_image(img_path, verify):
    """Trả về (w, h) hoặc lý do lỗi dạng chuỗi."""
    try:
        if img_path.stat().st_size == 0:
            return "zero_byte_image"
        with Image.open(img_path) as im:
            if verify == "full":
                im.load()  # giải mã toàn bộ pixel -> bắt được ảnh bị cắt cụt
            else:
                im.verify()  # chỉ kiểm tra cấu trúc file (nhanh)
            w, h = im.size
    except Exception:
        return "corrupted_image"
    if w <= 0 or h <= 0:
        return "corrupted_image"
    return w, h


def parse_label_file(lbl_path, img_w, img_h, args, stats):
    """Đọc 1 file nhãn YOLO, trả về (boxes_mới, số box mục tiêu bị loại vì lỗi)."""
    boxes, seen, rejected_target = [], set(), 0
    text = lbl_path.read_text(encoding="utf-8-sig", errors="replace")  # utf-8-sig: bỏ BOM đầu file
    for raw in text.splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            cls_f = float(parts[0])
            if not math.isfinite(cls_f) or cls_f != int(cls_f) or cls_f < 0:
                raise ValueError
            cls = int(cls_f)
        except ValueError:
            # Không đọc được class -> có thể là box mục tiêu: ảnh này không được làm ảnh nền
            stats["line_malformed"] += 1
            rejected_target += 1
            continue
        try:
            coords = [float(v) for v in parts[1:]]
            if not all(math.isfinite(v) for v in coords):
                raise ValueError
        except ValueError:
            stats["line_malformed"] += 1
            rejected_target += cls in CLASS_MAP
            continue

        # --- lọc lớp & đánh lại chỉ mục ---
        if cls in DROP_CLASSES:
            stats[f"drop_class_{cls}"] += 1
            continue
        if cls not in CLASS_MAP:
            stats[f"drop_unknown_class_{cls}"] += 1
            continue
        new_cls = CLASS_MAP[cls]

        # --- bbox hoặc polygon ---
        if len(coords) == 4:
            x, y, w, h = coords
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs, ys = coords[0::2], coords[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            stats["polygon_to_bbox"] += 1
        else:
            stats["line_malformed"] += 1
            rejected_target += 1
            continue

        # --- kiểm tra tính hợp lệ ---
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            stats["box_center_out_of_bounds"] += 1
            rejected_target += 1
            continue
        if w <= 0 or h <= 0:
            stats["box_non_positive_size"] += 1
            rejected_target += 1
            continue
        if w > 1.0 or h > 1.0:  # không phải tọa độ chuẩn hóa (vd đơn vị pixel)
            stats["box_not_normalized"] += 1
            rejected_target += 1
            continue
        overflow = max(w / 2 - x, x + w / 2 - 1.0, h / 2 - y, y + h / 2 - 1.0)
        if overflow > args.max_overflow:  # tràn biên quá nhiều -> nhãn lỗi, không clip
            stats["box_out_of_bounds"] += 1
            rejected_target += 1
            continue

        # Cắt phần tràn biên nhỏ về [0, 1]
        x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
        x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
        if overflow > 0:
            stats["box_clipped"] += 1
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

        w_px, h_px = w * img_w, h * img_h
        if min(w_px, h_px) <= 0 or w_px < args.min_box_px or h_px < args.min_box_px or round(min(w, h), 6) <= 0:
            stats["box_too_small"] += 1
            rejected_target += 1
            continue
        if args.max_aspect > 0 and max(w_px / h_px, h_px / w_px) > args.max_aspect:
            stats["box_abnormal_aspect"] += 1
            rejected_target += 1
            continue

        line = f"{new_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        if line in seen:
            stats["box_duplicate"] += 1
            continue
        seen.add(line)
        boxes.append(line)
    return boxes, rejected_target


def process_sample(sample, args):
    stats = Counter()
    res = check_image(sample["img"], args.verify)
    if isinstance(res, str):
        stats[res] += 1
        return {**sample, "status": res, "boxes": [], "stats": stats}
    img_w, img_h = res
    sample = {**sample, "w": img_w, "h": img_h}

    if sample["lbl"] is None:
        # Roboflow không xuất file .txt cho ảnh "null" (ảnh nền đã duyệt)
        status = "background" if args.missing_labels == "background" else "missing_label"
        return {**sample, "status": status, "bg_kind": "no_label_file", "boxes": [], "stats": stats}

    try:
        boxes, rejected = parse_label_file(sample["lbl"], img_w, img_h, args, stats)
    except Exception:  # 1 file lỗi lạ không được làm sập cả tiến trình
        stats["label_unreadable"] += 1
        return {**sample, "status": "label_unreadable", "boxes": [], "stats": stats}

    if boxes:
        status = "positive"
    elif rejected:
        # Ảnh có đối tượng mục tiêu nhưng box lỗi -> KHÔNG được dùng làm ảnh nền
        status = "tainted_background"
    else:
        status = "background"
    return {**sample, "status": status, "bg_kind": "annotated_negative", "boxes": boxes, "stats": stats}


# --------------------------------------------------------------------------- #
# Bước 5: gom nhóm theo chuỗi frame & chia tập phân tầng
# --------------------------------------------------------------------------- #
def source_stem(file_name):
    """'frame_12_jpg.rf.<hash>.jpg' -> 'frame_12_jpg' (bỏ hậu tố augment của Roboflow)."""
    return ROBOFLOW_SUFFIX.sub("", Path(file_name).stem)


def sequence_key(stem):
    """Họ chuỗi (mọi cụm số -> '#') và bộ số dùng để sắp thứ tự ảnh trong họ.

    'frame0-00-12-05_jpg' -> ('frame#-#-#-#_jpg', (0, 0, 12, 5)): mọi frame của một phiên quay rơi vào
    cùng một họ và được sắp đúng thứ tự thời gian, kể cả khi tên chứa nhiều trường số (giờ-phút-giây).
    """
    clean = HEX_HASH.sub("", stem)  # bỏ hash dài để không lấy nhầm số trong hash
    return DIGITS.sub("#", clean), tuple(int(n) for n in DIGITS.findall(clean))


def assign_groups(records, group_size):
    """Nhóm = mọi bản augment Roboflow của một ảnh gốc + group_size ảnh gốc liên tiếp cùng họ chuỗi."""
    families = defaultdict(set)
    for r in records:
        r["source"] = source_stem(r["img"].name)
        r["family"], order = sequence_key(r["source"])
        families[r["family"]].add((order, r["source"]))
    position = {}
    for family in families:
        for idx, (_, stem) in enumerate(sorted(families[family])):
            position[stem] = idx
    for r in records:
        r["seq_pos"] = position[r["source"]]
        r["group"] = f"{r['family']}@{r['seq_pos'] // max(1, group_size)}"
    return records


def stratified_group_split(records, ratios, seed, max_passes=50):
    """Chia nhóm vào train/val/test sao cho 5 chỉ số [box sign, box ped, ảnh, ảnh nền, số nhóm]
    của mỗi split bám sát tỷ lệ mục tiêu (số nhóm đảm bảo val/test đủ đa dạng cảnh quay).

    Hàm mục tiêu: tổng bình phương độ lệch TƯƠNG ĐỐI (frac - r) / r trên mọi split và chỉ số
    (chuẩn hóa theo r để lệch 1% ở tập test 10% bị phạt nặng như lệch 8% ở tập train 80%).
      1. Greedy: duyệt nhóm từ lớn đến nhỏ, đặt vào split làm chi phí tăng ít nhất.
      2. Tìm kiếm cục bộ (vector hóa numpy): với từng nhóm, chọn phép di chuyển sang split khác
         hoặc hoán đổi với một nhóm của split khác giảm chi phí nhiều nhất; lặp tới khi hội tụ.
    """
    names = sorted({r["group"] for r in records})
    index = {g: i for i, g in enumerate(names)}
    vec = np.zeros((len(names), 5))
    vec[:, 4] = 1
    for r in records:
        i = index[r["group"]]
        for line in r["boxes"]:
            vec[i, int(line[0])] += 1
        vec[i, 2] += 1
        vec[i, 3] += not r["boxes"]
    total = np.maximum(vec.sum(0), 1)
    target = np.array([ratios[s] for s in OUT_SPLITS])

    def row_cost(counts, r):
        """Chi phí của các hàng counts (..., 5) thuộc split có tỷ lệ r (vô hướng hoặc mảng)."""
        r = np.asarray(r)[..., None]
        return (((counts / total - r) / r) ** 2).sum(-1)

    order = np.random.default_rng(seed).permutation(len(names))  # phá thế hòa một cách tái lập được
    order = order[np.argsort(-(vec[order] / total).sum(1), kind="stable")]

    current = np.zeros((len(OUT_SPLITS), 5))
    assign = np.zeros(len(names), dtype=int)
    for i in order:
        s = int(np.argmin(row_cost(current + vec[i], target) - row_cost(current, target)))
        assign[i] = s
        current[s] += vec[i]

    for _ in range(max_passes):
        improved = False
        for i in order:
            a = assign[i]
            base = row_cost(current, target)
            # Di chuyển nhóm i: a -> b
            move = row_cost(current[a] - vec[i], target[a]) + row_cost(current + vec[i], target) - base[a] - base
            move[a] = np.inf
            # Hoán đổi nhóm i (ở a) với nhóm j (ở b != a)
            others = np.nonzero(assign != a)[0]
            swap_j, swap_gain = None, np.inf
            if len(others):
                b = assign[others]
                delta = vec[i] - vec[others]
                gain = (row_cost(current[a] - delta, target[a]) + row_cost(current[b] + delta, target[b])
                        - base[a] - base[b])
                k = int(np.argmin(gain))
                swap_j, swap_gain = others[k], gain[k]
            if min(move.min(), swap_gain) >= -1e-12:
                continue
            improved = True
            if move.min() <= swap_gain:
                b = int(np.argmin(move))
                current[a] -= vec[i]
                current[b] += vec[i]
                assign[i] = b
            else:
                b = assign[swap_j]
                delta = vec[i] - vec[swap_j]
                current[a] -= delta
                current[b] += delta
                assign[i], assign[swap_j] = b, a
        if not improved:
            break

    for r in records:
        r["split"] = OUT_SPLITS[assign[index[r["group"]]]]
    return records


def _near(sorted_positions, pos, k):
    i = bisect.bisect_left(sorted_positions, pos - k)
    return i < len(sorted_positions) and sorted_positions[i] <= pos + k


def count_adjacent_leaks(records, k):
    """Số ảnh gốc val/test có ảnh gốc cùng họ chuỗi cách <= k vị trí nằm trong train."""
    train_pos, eval_items = defaultdict(list), set()
    for r in records:
        if r["split"] == "train":
            train_pos[r["family"]].append(r["seq_pos"])
        else:
            eval_items.add((r["family"], r["seq_pos"]))
    for positions in train_pos.values():
        positions.sort()
    return sum(_near(train_pos[family], pos, k) for family, pos in eval_items)


def apply_embargo(records, k):
    """Loại khỏi train các ảnh gốc cách một ảnh gốc val/test cùng họ chuỗi <= k vị trí (purging),
    để các frame gần như trùng lặp ở ranh giới nhóm không rò rỉ sang tập đánh giá."""
    if k <= 0:
        return records, 0
    eval_pos = defaultdict(list)
    for r in records:
        if r["split"] != "train":
            eval_pos[r["family"]].append(r["seq_pos"])
    for positions in eval_pos.values():
        positions.sort()
    kept = [r for r in records if r["split"] != "train" or not _near(eval_pos.get(r["family"], []), r["seq_pos"], k)]
    return kept, len(records) - len(kept)


# --------------------------------------------------------------------------- #
# Bước 6: ghi kết quả
# --------------------------------------------------------------------------- #
def place_file(src, dst, mode):
    if mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:  # Windows không bật Developer Mode -> thử hardlink rồi sao chép
            mode = "hardlink"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # khác ổ đĩa / FS không hỗ trợ -> sao chép
    shutil.copyfile(src, dst)  # không chép thuộc tính read-only của file gốc


def remove_tree(path):
    """shutil.rmtree chịu được file read-only (Windows)."""
    def on_error(func, p, _):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=on_error)
    else:
        shutil.rmtree(path, onerror=on_error)


def write_outputs(records, dst, args):
    for s in OUT_SPLITS:
        (dst / "images" / s).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / s).mkdir(parents=True, exist_ok=True)

    # Tên nhãn = stem ảnh + .txt -> phải duy nhất theo stem (không phân biệt hoa thường),
    # nếu không a.jpg và a.png sẽ ghi đè cùng một file a.txt
    used = set()
    for r in records:
        name, k = r["img"].name, 0
        while Path(name).stem.lower() in used:  # trùng tên giữa các split gốc -> thêm tiền tố
            k += 1
            name = f"{r['origin']}{k if k > 1 else ''}_{r['img'].name}"
        used.add(Path(name).stem.lower())
        r["out_name"] = name

    def write_one(r):
        img_dst = dst / "images" / r["split"] / r["out_name"]
        place_file(r["img"], img_dst, args.link_mode)
        lbl_dst = dst / "labels" / r["split"] / (Path(r["out_name"]).stem + ".txt")
        lbl_dst.write_text("".join(f"{b}\n" for b in r["boxes"]), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in progress(ex.map(write_one, records), len(records), "Ghi dữ liệu"):
            pass

    # Không ghi khóa `path`: Ultralytics lấy thư mục chứa data.yaml làm gốc dataset, nên thư mục
    # processed/ chép từ Windows sang server Linux (hoặc đổi chỗ) vẫn dùng được nguyên trạng.
    yaml_text = (
        "# Sinh tự động bởi data_engineering/clean_and_filter.py - chạy lại script thay vì sửa tay.\n"
        "# Gốc dataset = thư mục chứa file này.\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        f"nc: {len(NEW_NAMES)}\n"
        "names:\n" + "".join(f"  {k}: {v}\n" for k, v in NEW_NAMES.items())
    )
    (dst / "data.yaml").write_text(yaml_text, encoding="utf-8")

    with open(dst / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "file", "source_path", "origin_split", "group", "width", "height",
                         "n_traffic_sign", "n_pedestrian"])
        for r in records:
            n = Counter(int(b[0]) for b in r["boxes"])
            writer.writerow([r["split"], r["out_name"], r["rel"], r["origin"], r["group"],
                             r["w"], r["h"], n[0], n[1]])


# --------------------------------------------------------------------------- #
# Báo cáo
# --------------------------------------------------------------------------- #
def split_summary(records):
    summary = {s: {"images": 0, "background": 0, "traffic_sign": 0, "pedestrian": 0, "groups": set(),
                   "sources": set()} for s in OUT_SPLITS}
    for r in records:
        row = summary[r["split"]]
        row["images"] += 1
        row["background"] += int(not r["boxes"])
        row["groups"].add(r["group"])
        row["sources"].add(r["source"])
        for b in r["boxes"]:
            row[NEW_NAMES[int(b[0])]] += 1
    for row in summary.values():
        row["groups"], row["sources"] = len(row["groups"]), len(row["sources"])
    return summary


def print_summary(summary):
    keys = ("images", "background", "traffic_sign", "pedestrian", "groups")
    totals = {k: sum(summary[s][k] for s in OUT_SPLITS) for k in keys}
    print(f"\n{'split':<7}" + "".join(f"{k:>20}" for k in keys))
    for s in OUT_SPLITS:
        cells = "".join(
            f"{summary[s][k]:>11} ({100 * summary[s][k] / max(1, totals[k]):5.1f}%)" for k in keys)
        print(f"{s:<7}{cells}")
    print(f"{'total':<7}" + "".join(f"{totals[k]:>20}" for k in keys))
    if totals["pedestrian"]:
        print(f"\nTỷ lệ mất cân bằng traffic_sign : pedestrian = "
              f"{totals['traffic_sign'] / totals['pedestrian']:.2f} : 1")
    for s in OUT_SPLITS:
        for cls in NEW_NAMES.values():
            if summary[s][cls] == 0:
                print(f"[CẢNH BÁO] Tập {s} không có box nào của lớp {cls}!")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Lọc 2 lớp, làm sạch và chia tập dữ liệu YOLO (BKAuto).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--src", type=Path, default=Path("datasets/raw"), help="Thư mục dataset gốc (YOLO)")
    p.add_argument("--dst", type=Path, default=Path("datasets/processed"), help="Thư mục đầu ra")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--bg-ratio", type=float, default=0.10,
                   help="Ảnh nền chiếm tỷ lệ này trong TỔNG số ảnh đầu ra (khuyến nghị 5-10%%); "
                        "phần ảnh nền dư bị loại. 0 = bỏ hết ảnh nền")
    p.add_argument("--missing-labels", choices=("background", "skip"), default="background",
                   help="Ảnh không có file .txt: 'background' = coi là ảnh nền (ảnh null của Roboflow), "
                        "'skip' = loại bỏ")
    p.add_argument("--group-size", type=int, default=20,
                   help="Số ảnh gốc liên tiếp của một chuỗi gom thành 1 nhóm khi chia tập")
    p.add_argument("--embargo", type=int, default=5,
                   help="Loại khỏi train các ảnh gốc cách ảnh val/test cùng chuỗi <= N vị trí (0 = tắt)")
    p.add_argument("--min-box-px", type=float, default=2.0, help="Cạnh box nhỏ nhất (pixel)")
    p.add_argument("--max-aspect", type=float, default=20.0,
                   help="Tỷ lệ cạnh dài/cạnh ngắn tối đa của box (0 = tắt)")
    p.add_argument("--max-overflow", type=float, default=0.05,
                   help="Phần box tràn ra ngoài ảnh tối đa (tỷ lệ theo cạnh ảnh) còn được cắt về biên; "
                        "tràn nhiều hơn -> loại box")
    p.add_argument("--verify", choices=("full", "header"), default="full",
                   help="full = giải mã toàn bộ ảnh, header = chỉ kiểm tra cấu trúc (nhanh)")
    p.add_argument("--link-mode", choices=("copy", "hardlink", "symlink"), default="copy")
    p.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4) * 2))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true",
                   help="Ghi đè --dst nếu đó là đầu ra cũ của script này")
    p.add_argument("--dry-run", action="store_true", help="Chỉ thống kê, không ghi file")
    args = p.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6 or min(args.ratios) <= 0:
        p.error("--ratios phải dương và có tổng bằng 1")
    if not 0.0 <= args.bg_ratio < 1.0:
        p.error("--bg-ratio phải nằm trong [0, 1)")
    if args.group_size < 1 or args.embargo < 0 or args.workers < 1:
        p.error("--group-size, --workers phải >= 1 và --embargo >= 0")
    if args.min_box_px < 0 or args.max_aspect < 0 or args.max_overflow < 0:
        p.error("--min-box-px, --max-aspect, --max-overflow không được âm")
    return args


def is_previous_output(path):
    return (path / "data.yaml").is_file() and (path / "split_manifest.csv").is_file()


def check_destination(src, dst, args):
    """Chặn mọi trường hợp có thể xóa nhầm dữ liệu gốc. Trả về True nếu cần xóa dst trước khi ghi."""
    if dst == src or is_relative_to(src, dst):
        sys.exit("[LỖI] --dst không được trùng hoặc chứa --src (tránh xóa nhầm dữ liệu gốc).")
    if not dst.exists() or not any(dst.iterdir()):
        return False
    if not is_previous_output(dst):
        sys.exit(f"[LỖI] {dst} đã có dữ liệu nhưng không phải đầu ra của script này - từ chối ghi đè/xóa.")
    if not args.overwrite and not args.dry_run:
        sys.exit(f"[LỖI] {dst} đã tồn tại. Thêm --overwrite để ghi đè.")
    return not args.dry_run


def main():
    for stream in (sys.stdout, sys.stderr):  # tránh UnicodeEncodeError trên console Windows
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()
    ratios = dict(zip(OUT_SPLITS, args.ratios))
    if not src.is_dir():
        sys.exit(f"[LỖI] Không tìm thấy thư mục dataset gốc: {src}")
    must_clear_dst = check_destination(src, dst, args)

    # ---- In bảng ánh xạ lớp ----
    src_names = read_source_names(src)
    print("Ánh xạ lớp:")
    for c in sorted(set(CLASS_MAP) | DROP_CLASSES):
        target = f"-> {CLASS_MAP[c]} ({NEW_NAMES[CLASS_MAP[c]]})" if c in CLASS_MAP else "-> XÓA"
        print(f"  {c:>2} {src_names.get(c, '?'):<22} {target}")

    # ---- Bước 1 ----
    samples, orphan_labels = discover_samples(src, dst if is_relative_to(dst, src) else None)
    if not samples:
        sys.exit(f"[LỖI] Không tìm thấy ảnh nào trong {src}")
    print(f"\nTìm thấy {len(samples)} ảnh ({sum(s['lbl'] is not None for s in samples)} có file nhãn, "
          f"{orphan_labels} file nhãn mồ côi không có ảnh).")

    # ---- Bước 2 + 3 ----
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        records = list(progress(ex.map(lambda s: process_sample(s, args), samples), len(samples),
                                "Kiểm tra ảnh & nhãn"))
    stats = Counter()
    for r in records:
        stats.update(r["stats"])
    status = Counter(r["status"] for r in records)

    # ---- Bước 4: ảnh nền ----
    positives = [r for r in records if r["status"] == "positive"]
    if not positives:
        sys.exit("[LỖI] Không còn ảnh nào chứa traffic_sign/pedestrian - kiểm tra lại cấu trúc --src.")
    bg_pool = [r for r in records if r["status"] == "background"]
    n_bg_target = round(args.bg_ratio * len(positives) / (1.0 - args.bg_ratio))
    n_bg = min(len(bg_pool), n_bg_target)
    backgrounds = random.Random(args.seed).sample(bg_pool, n_bg)
    kept = sorted(positives + backgrounds, key=lambda r: r["rel"])
    if n_bg < n_bg_target:
        print(f"[CẢNH BÁO] Chỉ có {len(bg_pool)} ảnh nền khả dụng, thấp hơn tỷ lệ --bg-ratio mong muốn.")

    # ---- Bước 5: chia tập theo nhóm + embargo chống rò rỉ ----
    assign_groups(kept, args.group_size)
    stratified_group_split(kept, ratios, args.seed)
    leaks_before = {k: count_adjacent_leaks(kept, k) for k in (1, 5)}
    kept, purged = apply_embargo(kept, args.embargo)
    leaks_after = {k: count_adjacent_leaks(kept, k) for k in (1, 5)}
    summary = split_summary(kept)

    print("\nTrạng thái ảnh đầu vào:")
    for k, v in sorted(status.items()):
        print(f"  {k:<22}{v:>8}")
    bg_kinds = Counter(r["bg_kind"] for r in backgrounds)
    print(f"  -> giữ {len(positives)} ảnh có nhãn + {n_bg}/{len(bg_pool)} ảnh nền ({dict(bg_kinds)})")
    print("\nThống kê làm sạch nhãn:")
    for k, v in sorted(stats.items()):
        print(f"  {k:<28}{v:>8}")
    n_eval = sum(summary[s]["sources"] for s in OUT_SPLITS if s != "train")
    print(f"\nChống rò rỉ: {len({r['family'] for r in kept})} họ chuỗi, nhóm {args.group_size} ảnh gốc liên tiếp, "
          f"embargo ±{args.embargo} -> loại {purged} ảnh train giáp ranh val/test.")
    for k in (1, 5):
        print(f"  ảnh gốc val/test có ảnh train cùng chuỗi trong ±{k} frame: "
              f"{leaks_before[k]} -> {leaks_after[k]} / {n_eval}")
    print_summary(summary)

    empty = [s for s in OUT_SPLITS if summary[s]["images"] == 0]
    if empty:
        sys.exit(f"[LỖI] Tập {empty} rỗng - quá ít nhóm để chia. Giảm --group-size hoặc --embargo.")

    report = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "class_map": {str(k): v for k, v in CLASS_MAP.items()},
        "dropped_classes": sorted(DROP_CLASSES),
        "names": NEW_NAMES,
        "source_class_names": src_names,
        "input_images": len(samples),
        "orphan_label_files": orphan_labels,
        "image_status": dict(status),
        "kept": {"positive": len(positives), "background": n_bg, "background_pool": len(bg_pool),
                 "background_kinds": dict(bg_kinds), "embargo_purged_train_images": purged},
        "cleaning_stats": dict(stats),
        "leakage": {"adjacent_eval_sources_before_embargo": leaks_before,
                    "adjacent_eval_sources_after_embargo": leaks_after, "eval_sources": n_eval},
        "splits": summary,
    }

    if args.dry_run:
        print("\n[DRY-RUN] Không ghi file nào.")
        return

    # ---- Bước 6 ----
    if must_clear_dst:
        remove_tree(dst)
    write_outputs(kept, dst, args)
    (dst / "clean_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nHoàn tất! Dữ liệu đã lưu tại: {dst}")
    print(f"  - {dst / 'data.yaml'}\n  - {dst / 'split_manifest.csv'}\n  - {dst / 'clean_report.json'}")


if __name__ == "__main__":
    main()
