# Edge Deployment - YOLO26m trên Raspberry Pi 5 + Hailo-8

Hướng dẫn từ cài đặt HailoRT, build ứng dụng C++ thời gian thực đến chạy bộ đo kiểm tự động cho
báo cáo. Mọi lệnh chạy **trên Raspberry Pi 5**, từ thư mục gốc repo (`~/bkauto-yolov8-vision`), trừ khi
ghi chú khác.

```
edge_deployment/
├── inference_app.cpp          # App C++17 3 luồng: Capture -> Hailo-8 (async) -> Post-process/Display
├── include/detection_utils.hpp# Hàng đợi thread-safe, decode YOLO26 end-to-end, letterbox, thống kê (C++17 thuần)
├── tests/test_detection_utils.cpp
├── CMakeLists.txt / Makefile
├── benchmark_profiler.py      # Đo mAP / NPU / độ trễ / RAM / CPU / nhiệt độ -> CSV + JSON
└── requirements.txt
```

## 1. Kiến trúc ứng dụng C++

```
 Camera/Video ─► [Luồng 1: Capture]  cap.read() ─► letterbox 640 + BGR→RGB ghi thẳng vào buffer DMA
                        │  BoundedQueue  (camera: bỏ frame cũ nhất │ video: chờ, không bỏ frame)
                        ▼
                 [Luồng 2: Inference]  wait_for_async_ready ─► run_async (nhiều job song song trên NPU)
                        │  callback HailoRT đẩy frame đã xong
                        ▼
                 [Luồng 3: Post-process & Display = luồng chính]
                        decode YOLO26 (ltrb×stride, sigmoid, top-300, KHÔNG NMS) ─► toạ độ ảnh gốc ─► vẽ box + HUD
                        (FPS, E2E latency, NPU/Pre/Post ms, nhiệt độ SoC/NPU, số frame bỏ) ─► imshow/ghi video/CSV
```

- **Không copy thừa**: ảnh letterbox được ghi trực tiếp vào buffer căn trang đã `DmaMappedBuffer` map
  sẵn vào Hailo-8 (pool cố định, tái sử dụng).
- **Độ trễ thấp với camera**: bỏ frame cũ ở CẢ hai hàng đợi - hàng đợi vào (khi NPU chậm hơn camera) và
  hàng đợi hiển thị (luồng 3 chỉ vẽ frame mới nhất khi render chậm). Mô phỏng với camera 30 FPS, render
  45 ms/frame: độ trễ E2E 543 ms -> 90 ms.
- **Watchdog**: không có frame mới trong `--stall-timeout` giây (frame đầu: 10 s) -> báo lỗi, dừng có trật tự.
  Nếu `cap.read()` kẹt trong driver, app dừng NPU rồi buộc thoát sau 3 s; Ctrl+C lần 2 thoát ngay.
- **An toàn bộ nhớ**: `shared_ptr` giữ buffer sống tới khi callback của NPU chạy xong; slot tự trả về pool
  khi frame bị huỷ; `ConfiguredInferModel::shutdown()` trước khi huỷ pool; lỗi ở mọi luồng được gom về
  luồng chính bằng `std::exception_ptr`.
- Luồng 3 là luồng chính vì GUI OpenCV (GTK/Qt) chỉ an toàn khi mọi lệnh highgui chạy trên một luồng.
- HEF YOLO26 (NMS-free) từ `optimization/compile_npu.py`: đầu vào uint8 RGB NHWC 640×640; **6 đầu ra thô**
  của nhánh one2one, HailoRT trả FLOAT32 NHWC: box `(H, W, 4)` = khoảng cách ltrb theo stride (không DFL) và
  cls `(H, W, nc)` = logit, với stride 8/16/32 (80×80, 40×40, 20×20). App tự ghép cặp theo shape.
- Hậu xử lý trên CPU (`edge::decode_end2end`): `xyxy = (anchor ∓ ltrb) × stride`, `score = sigmoid(logit)`,
  giữ top-`--max-det` (300) cặp (anchor, lớp) có score > `--conf` - cho cùng kết quả với `Detect.postprocess`
  của Ultralytics; không có NMS vì nhánh one2one chỉ sinh một dự đoán cho mỗi vật thể. Lọc theo ngưỡng logit
  trước nên chỉ tính `exp()` cho các ô vượt ngưỡng (~1 ms cho 8400 anchor).
- TopK/Gather của đồ thị ONNX đầy đủ là phép phụ thuộc dữ liệu, không ánh xạ được lên Hailo-8, vì vậy HEF
  dừng ở 6 lớp Conv (giống exporter Hailo chính thức của Ultralytics cho YOLO26).

## 2. Chuẩn bị phần cứng & hệ điều hành

1. Raspberry Pi OS **64-bit** (Bookworm hoặc Trixie), Pi 5 + AI HAT+ 26 TOPS / M.2 HAT+ với module Hailo-8.
   Nên có tản nhiệt chủ động (Active Cooler) để số đo nhiệt độ/throttling ổn định.
2. PCIe Gen 3 cho băng thông tối đa tới NPU:
   ```bash
   sudo raspi-config        # Advanced Options -> PCIe Speed -> Yes (Gen 3), rồi reboot
   sudo lspci -vv | grep -E "Hailo|LnkSta:"   # LnkSta phải báo Speed 8GT/s
   ```

## 3. Cài HailoRT (thư viện + header + Python) trên Pi

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y hailo-all          # HailoRT, driver PCIe (dkms), firmware, python3-hailort, TAPPAS
sudo reboot
```

> Không có gói `hailort-dev` riêng: gói `hailort` (cài kèm `hailo-all`) đã chứa header
> `/usr/include/hailo/`, thư viện `/usr/lib/libhailort.so` và CMake config `/usr/lib/cmake/HailoRT/`.
> Phiên bản trên apt: Bookworm → HailoRT **4.20**, Trixie → HailoRT **4.23**.

Kiểm tra:
```bash
hailortcli fw-control identify          # thấy "Device Architecture: HAILO8"
hailortcli --version
ls /usr/include/hailo/hailort.hpp /usr/lib/cmake/HailoRT/HailoRTConfig.cmake
python3 -c "import hailo_platform; print(hailo_platform.__version__)"
```

### 3.1. Kiểm tra HEF có chạy được với HailoRT trên Pi (quan trọng)

Mỗi đợt Hailo AI Software Suite ghép một bản DFC với một bản HailoRT (3.30↔4.20, 3.31↔4.21, 3.32↔4.22,
3.33↔4.23, 3.34↔4.24). HEF từ DFC mới hơn HailoRT trên Pi có thể vẫn nạp được nếu không dùng tính năng mới
(HEF YOLOv8n trước đây từ DFC 3.34 chỉ dùng extension có từ HailoRT 4.18), nhưng cần kiểm tra trước khi build:

```bash
hailortcli parse-hef models/best_hailo.hef      # YOLO26: 1 input (640x640x3) + 6 output (80/40/20, 4 hoặc nc kênh)
hailortcli run models/best_hailo.hef            # chạy thử với dữ liệu ngẫu nhiên, in FPS
```

Nếu báo lỗi phiên bản HEF / "not supported", chọn MỘT trong hai cách:
- **Biên dịch lại cho khớp Pi** (khuyến nghị): trên server, cài DFC cùng đợt với HailoRT của Pi
  (Bookworm 4.20 → DFC 3.30; Trixie 4.23 → DFC 3.33) rồi chạy lại `optimization/compile_npu.py`.
- **Nâng HailoRT trên Pi** lên đúng bản đi cùng DFC 3.34 (tải `hailort_<ver>_arm64.deb`,
  `hailort-pcie-driver_<ver>_all.deb`, `python3-hailort` từ Hailo Developer Zone; gỡ `hailo-all` trước để
  tránh lẫn phiên bản driver/firmware).

## 4. Build ứng dụng C++

```bash
sudo apt install -y build-essential cmake pkg-config libopencv-dev \
                    gstreamer1.0-libcamera v4l-utils   # libcamera: camera CSI; v4l-utils: camera USB

cmake -S edge_deployment -B edge_deployment/build -DCMAKE_BUILD_TYPE=Release
cmake --build edge_deployment/build -j4
# hoặc:  make -C edge_deployment

# Unit test (không cần Hailo-8)
make -C edge_deployment test
```

CMake tìm `HailoRT` (>= 4.20, target `HailoRT::libhailort`) và OpenCV (`core imgproc imgcodecs videoio
highgui`), build C++17 Release với `-mcpu=native` (Cortex-A76) và `-Wall -Wextra -Wpedantic`.

## 5. Chạy ứng dụng

```bash
APP=edge_deployment/build/inference_app

# Camera USB (V4L2, MJPG 1280x720@30)
$APP --hef models/best_hailo.hef --source 0

# Camera CSI của Pi (libcamera qua GStreamer)
$APP --hef models/best_hailo.hef --source libcamera --width 1280 --height 720 --fps 30

# Video thử nghiệm, ghi lại video có overlay cho demo (1080p)
$APP --hef models/best_hailo.hef --source test_video.mp4 --save demo_out.mp4

# Chạy qua SSH không màn hình + ghi độ trễ từng frame
$APP --source test_video.mp4 --headless --stats frame_stats.csv --duration 60
```

| Tham số | Ý nghĩa (mặc định) |
|---|---|
| `--hef PATH` | HEF đã biên dịch (`models/best_hailo.hef`) |
| `--source SRC` | `0`, `/dev/video0`, `libcamera`, file video, hoặc pipeline GStreamer có `!` |
| `--width/--height/--fps` | Độ phân giải camera yêu cầu (1280×720@30) |
| `--conf` / `--max-det` | Ngưỡng tin cậy hiển thị (0.5) / số dự đoán tối đa - top-k của YOLO26 (300) |
| `--headless` | Không mở cửa sổ (SSH, đo kiểm) |
| `--save out.mp4` | Ghi video kết quả có overlay |
| `--stats file.csv` | Ghi `preprocess/queue/npu/postprocess/e2e` ms, số box, nhiệt độ từng frame |
| `--max-frames N`, `--duration S`, `--loop` | Giới hạn số frame / thời gian, lặp lại video |
| `--queue N`, `--drop` / `--no-drop` | Hàng đợi đầu vào (4); bỏ frame cũ ở cả 2 hàng đợi (mặc định: camera bỏ, video chờ - đo kiểm không mất frame) |
| `--stall-timeout S` | Báo lỗi & thoát nếu không có frame mới trong S giây (3) |

Phím `q`/`Esc` hoặc `Ctrl+C` để dừng; khi kết thúc app in bảng mean/p50/p95/p99 của từng tầng.
Mã thoát: `0` bình thường · `1` lỗi (mất camera, watchdog, HailoRT) · `3` buộc thoát vì luồng camera treo ·
`130` Ctrl+C lần hai.
Chạy cửa sổ qua SSH: `export DISPLAY=:0` (hiển thị lên màn hình gắn với Pi) hoặc dùng `--headless`.

**HUD**: FPS tức thời (cửa sổ 1 s) · E2E latency (từ lúc nhận frame tới khi có kết quả đã vẽ) ·
NPU ms (từ lúc gửi job tới callback, gồm truyền PCIe + giải lượng tử) · Pre/Post ms (Post = decode +
top-k + vẽ) · nhiệt độ SoC/NPU
· số frame bị bỏ.

## 6. Đo kiểm tự động (benchmark_profiler.py)

### 6.1. Môi trường Python

```bash
sudo apt install -y python3-numpy python3-opencv python3-psutil     # khuyến nghị: không cần pip
# Nếu dùng venv thì bắt buộc --system-site-packages để thấy hailo_platform:
#   python3 -m venv --system-site-packages ~/bkauto-venv && source ~/bkauto-venv/bin/activate
#   pip install -r edge_deployment/requirements.txt      # KHÔNG pip install numpy (xem file)
```

Chép tập Test từ server sang Pi (giữ cấu trúc `datasets/processed/{images,labels}/test`):
```bash
mkdir -p datasets/processed/images datasets/processed/labels
rsync -av server:~/bkauto-yolov8-vision/datasets/processed/images/test datasets/processed/images/
rsync -av server:~/bkauto-yolov8-vision/datasets/processed/labels/test datasets/processed/labels/
```

### 6.2. Các lệnh

```bash
B="python3 edge_deployment/benchmark_profiler.py"

$B accuracy --hef models/best_hailo.hef           # mAP toàn bộ tập Test qua NPU (conf 0.001, IoU 0.7)
$B accuracy --backend onnx --onnx models/best.onnx   # baseline FP32 (đầu ra end-to-end (1,300,6)), cần onnxruntime
$B npu --hef models/best_hailo.hef                # hailortcli benchmark: FPS hw_only + độ trễ NPU thuần
$B stress --engine cpp --source test_video.mp4    # 5 phút: đo chính inference_app (RAM, CPU, nhiệt độ)
$B stress --engine python                         # 5 phút: vòng lặp HailoRT Python trên ảnh Test

# Trọn bộ cho báo cáo (accuracy + baseline ONNX + NPU + stress 300 s)
$B all --hef models/best_hailo.hef --onnx models/best.onnx --engine cpp --source test_video.mp4
```

Kết quả ở `edge_deployment/results/<thời-gian>/`:

| File | Nội dung |
|---|---|
| `summary.csv` / `summary.json` | Bảng tổng hợp theo nhóm Độ chính xác / Độ trễ & Tốc độ / Tài nguyên |
| `accuracy_hailo.json`, `accuracy_hailo_per_class.csv` | mAP@0.5, mAP@0.5:0.95, P, R, F1 toàn cục + từng lớp |
| `accuracy_onnx.json` | Baseline FP32 (khi có `--onnx`) → ΔmAP = mAP_NPU − mAP_ONNX |
| `npu_benchmark.json/.txt` | FPS hw_only/streaming, Latency hw/overall từ `hailortcli` |
| `stress_<engine>.json`, `stress_<engine>_samples.csv` | Độ trễ mean/p50/p95/p99, FPS; mẫu mỗi giây: RSS, CPU tiến trình/hệ thống, RAM hệ thống, tần số CPU, nhiệt độ SoC & NPU |
| `app_frame_stats.csv`, `app_stdout.log` | Log từng frame của inference_app (engine cpp) |
| `benchmark.log`, `system_info.json` | Nhật ký chạy, thông tin bo mạch/HailoRT |

### 6.3. Ánh xạ sang bảng chỉ số của đề bài

| Chỉ số đề bài | Nguồn trong kết quả |
|---|---|
| mAP@0.5, mAP@0.5:0.95 (NPU) | `accuracy_hailo.json` - thuật toán khớp/AP port nguyên văn từ Ultralytics (nội suy 101 điểm COCO) |
| ΔmAP | `summary.csv` (`ΔmAP@…` = NPU − ONNX FP32, cùng tiền xử lý + cùng top-k end-to-end → chỉ còn sai số lượng tử hóa); có thể so thêm với mAP Test của baseline PyTorch |
| NPU Inference Time | `npu_benchmark.json → latency_hw_ms` (thuần phần cứng); `Inference call time` = thời gian gọi từ host |
| Preprocessing / Postprocessing / End-to-End | `stress_*.json → latency_ms` (YOLO26 không có NMS: Postprocessing = decode + sigmoid + top-k trên CPU) |
| Throughput (FPS) | `stress_*.json → fps` (thực tế) và `npu_benchmark.json → fps_hw_only` (trần của NPU) |
| Footprint `.pt` / `.onnx` / `.hef` | `summary.csv → Footprint …` |
| RAM | `Process RAM (RSS)` của tiến trình đo + `System RAM used` |
| CPU / NPU utilization | `CPU utilization (system/process)`; NPU ước lượng = FPS thực / FPS hw_only. Số đo trực tiếp: chạy app với `HAILO_MONITOR=1` rồi mở terminal khác chạy `hailortcli monitor` |
| Thermal (sau 5 phút) | `SoC temperature (sau stress)` = trung bình 10 s cuối, `NPU temperature (sau stress)`; `throttled` (`vcgencmd get_throttled`, `0x0` = không throttle) |

Lưu ý khi đo:
- mAP đo với `conf=0.001`, `max_det=300` (chuẩn Ultralytics val). HEF YOLO26 không chứa ngưỡng nào - ngưỡng
  chỉ áp dụng trong bước decode trên CPU nên đổi tự do lúc chạy.
- Để số đo tài nguyên sạch: tắt desktop/ứng dụng khác, đặt governor hiệu năng
  `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`, ghi rõ có/không
  có tản nhiệt trong báo cáo.
- Mỗi lúc chỉ một tiến trình dùng Hailo-8: không chạy đồng thời app C++ và script Python (lỗi
  `HAILO_OUT_OF_PHYSICAL_DEVICES`). Engine `cpp` của lệnh `stress` tự chạy app làm tiến trình con.

## 7. Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `Could not find a package configuration file provided by "HailoRT"` | `sudo apt install hailo-all`; hoặc thêm `-DHailoRT_DIR=/usr/lib/cmake/HailoRT` |
| `Không nạp được HEF` / lỗi phiên bản HEF | Xem mục 3.1 - DFC và HailoRT phải cùng đợt phát hành |
| `HAILO_OUT_OF_PHYSICAL_DEVICES` / device busy | Tiến trình khác đang giữ NPU: `sudo lsof /dev/hailo0`, đóng app/script khác |
| `HAILO_DRIVER_...` sau khi nâng kernel | `sudo apt install --reinstall hailo-dkms` (hoặc `hailort-pcie-driver`) rồi reboot |
| Camera CSI không mở được | `rpicam-hello --list-cameras`; cài `gstreamer1.0-libcamera`; thử `--source libcamera` |
| `ModuleNotFoundError: hailo_platform` trong venv | Tạo lại venv với `--system-site-packages` |
| FPS thấp hơn kỳ vọng | Kiểm tra PCIe Gen 3, governor `performance`, `--headless` (imshow tốn CPU), throttling |
