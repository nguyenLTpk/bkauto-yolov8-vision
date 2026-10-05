# BKAuto - YOLO26m Object Detection trên Raspberry Pi 5 + Hailo-8

Pipeline Edge AI end-to-end cho xe tự hành RC trên sa bàn: dữ liệu thô → YOLO26m (NMS-free)
(RTX 5070 Ti) → ONNX → Hailo DFC (`.hef`) → suy luận thời gian thực trên Raspberry Pi 5 + Hailo-8.

Hai lớp mục tiêu: `0: traffic_sign` (gộp 9 loại biển báo gốc), `1: pedestrian`.

## 0. Môi trường server

```bash
python -m venv .venv && source .venv/bin/activate
# PyTorch build CUDA 12.8 (bắt buộc cho GPU Blackwell sm_120 như RTX 5070 Ti)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Giải nén dataset gốc (Roboflow, định dạng YOLO) vào `datasets/raw/`. Script tự dò mọi thư mục con
`images/` + `labels/`, nên giữ nguyên cấu trúc `Training_Yolo_Format_UPDATE/data_set_for_training/...`.

## 1. Lọc lớp, làm sạch và chia tập

```bash
python data_engineering/clean_and_filter.py --src datasets/raw --dst datasets/processed
```

- Gộp class 0–8 → `traffic_sign`, class 9 → `pedestrian`; xóa box của class 10–16 (và class lạ, vd 17).
- Bỏ ảnh 0 byte / hỏng; bỏ box lỗi (tâm ngoài [0,1], w/h ≤ 0 hoặc > 1, tràn biên > 5%, NaN, quá nhỏ,
  tỷ lệ cạnh > 20); polygon → bbox; box tràn biên ít được cắt về biên.
- Ảnh nền chiếm 10% đầu ra. Ảnh có box mục tiêu bị lỗi không được dùng làm ảnh nền.
- Chia 80:10:10 theo nhóm frame liên tiếp (chống rò rỉ giữa các frame video), phân tầng theo
  số box từng lớp, số ảnh, số ảnh nền và số nhóm; embargo ±5 frame quanh ảnh val/test.
- Đầu ra: `datasets/processed/{images,labels}/{train,val,test}`, `data.yaml`,
  `split_manifest.csv` (truy vết từng ảnh), `clean_report.json` (số liệu cho báo cáo).
- `--dry-run` để chỉ xem thống kê; `--help` để xem toàn bộ tham số.

## 2. Huấn luyện baseline

```bash
python training/train.py                      # cấu hình: training/configs/yolo26m_baseline.yaml
python training/train.py --config training/configs/yolov8n_baseline.yaml   # YOLOv8n chỉ tham khảo -> models/yolov8n_best.pt (không deploy)
tensorboard --logdir runs/train               # theo dõi loss / mAP
```

YOLO26m (head end-to-end NMS-free, không DFL), `imgsz=640`, batch 16, AdamW + cosine LR, `patience=50`
(Early Stopping theo mAP@0.5:0.95 trên val),
image-weighted sampling để cân bằng `traffic_sign` ≫ `pedestrian`. Mô hình tốt nhất được chép ra
`models/best.pt`.

## 3. Xuất ONNX cho Hailo DFC

```bash
python optimization/export_onnx.py
```

- `models/best.onnx`: đồ thị đầy đủ (opset 11, tĩnh 1×3×640×640, đã simplify). YOLO26 (`nms=False` → nhánh
  one2one): đầu ra `(1, 300, 6)` = `[x1, y1, x2, y2, score, class]`, đã gồm TopK/GatherElements, không NMS.
- `models/best_hailo.onnx`: cắt tại 6 Conv cuối của nhánh one2one (box ltrb 4 kênh + logit lớp × 3 stride),
  không còn TopK/Gather; decode + sigmoid + top-300 chạy trên CPU. Script tự kiểm tra (đồ thị cắt + decode
  numpy) ≡ đồ thị đầy đủ. Pipeline chỉ hỗ trợ YOLO26 NMS-free: trọng số YOLOv8 bị từ chối ở bước export.
- `models/best_hailo_meta.json`: `end_node_names`, stride, chuẩn hóa đầu vào cho bước biên dịch.

## 4. Calibration set cho PTQ INT8

```bash
python optimization/calibrate_ptq.py          # 200 ảnh train -> models/calib/calib_set_640.npy
```

Chọn ảnh phân tầng theo lớp (≥ 25% có pedestrian), xoay vòng qua các chuỗi video, mỗi ảnh gốc 1 bản và
bỏ bản bị Roboflow augment mạnh (xoay, ảnh xám, nhiễu). Tiền xử lý giống hệt lúc suy luận: letterbox 640
(pad 114), RGB, NHWC, **uint8 0..255**. Phép chia 255 được gộp vào NPU (lớp `normalization` của Hailo).

## 5. Biên dịch HEF cho Hailo-8 (Linux / WSL2)

Hailo DFC chỉ chạy trên Linux x86_64 (≥ 16 GB RAM) - khuyến nghị chạy trên server; cài đặt theo
`optimization/requirements-hailo.txt`.

```bash
source ~/hailo_dfc/bin/activate
python optimization/compile_npu.py            # -> models/best_hailo.hef
python optimization/calibrate_ptq.py --num 1024 && python optimization/compile_npu.py --compiler-max  # server GPU
hailo profiler models/best_hailo_build/best_hailo_compiled.har    # báo cáo layer/hiệu năng
```

Parsing (6 end-node Conv) → PTQ INT8 (`compression_level=0`, 6 output giữ a16) → biên dịch. HEF YOLO26
nhận ảnh uint8 RGB 640×640 và trả 6 tensor thô; KHÔNG `nms_postprocess`, không sigmoid on-chip - decode +
top-k NMS-free do app trên Pi đảm nhận. Nhật ký, model script `.alls` và `compile_report.json`
nằm trong `models/best_hailo_build/`.
