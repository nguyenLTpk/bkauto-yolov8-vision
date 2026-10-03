# BKAuto - YOLOv8n Object Detection trên Raspberry Pi 5 + Hailo-8

Pipeline Edge AI end-to-end cho xe tự hành RC trên sa bàn: dữ liệu thô → YOLOv8n baseline
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
python training/train.py                      # cấu hình: training/configs/yolov8n_baseline.yaml
tensorboard --logdir runs/train               # theo dõi loss / mAP
```

YOLOv8n, `imgsz=640`, AdamW + cosine LR, `patience=50` (Early Stopping theo mAP@0.5:0.95 trên val),
image-weighted sampling để cân bằng `traffic_sign` ≫ `pedestrian`. Mô hình tốt nhất được chép ra
`models/best.pt`.

## 3. Xuất ONNX cho Hailo DFC

```bash
python optimization/export_onnx.py
```

- `models/best.onnx`: đồ thị đầy đủ (opset 11, tĩnh 1×3×640×640, đã simplify, không NMS).
- `models/best_hailo.onnx`: cắt tại 6 Conv cuối của Detect head (thuần Conv/SiLU/Concat/Resize/MaxPool),
  phần decode DFL + NMS chuyển sang Hailo `nms_postprocess` hoặc CPU.
- `models/best_hailo_meta.json`: `end_node_names`, stride, chuẩn hóa đầu vào cho bước biên dịch.
