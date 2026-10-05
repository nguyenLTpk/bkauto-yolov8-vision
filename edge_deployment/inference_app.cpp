// inference_app.cpp - Ứng dụng phát hiện vật thể thời gian thực trên Raspberry Pi 5 + Hailo-8.
//
// Kiến trúc Producer-Consumer 3 luồng:
//   [Luồng 1: Capture]  đọc camera/video -> letterbox + BGR->RGB ghi thẳng vào buffer DMA của NPU
//            | BoundedQueue<FramePtr> (camera: bỏ frame cũ nhất để giữ độ trễ thấp; video: chờ)
//   [Luồng 2: Inference] wait_for_async_ready -> run_async (HailoRT, nhiều job song song trên NPU)
//            | callback của HailoRT đẩy frame đã xong vào hàng đợi kết quả
//   [Luồng 3: Post-process & Display - luồng chính] decode YOLO26 end-to-end (NMS-free) -> toạ độ ảnh
//            gốc -> vẽ box, nhãn, HUD (FPS, độ trễ end-to-end, NPU, nhiệt độ) -> imshow / ghi video / CSV
// Luồng 3 là luồng chính vì GUI của OpenCV (GTK/Qt) chỉ an toàn khi mọi lệnh highgui chạy trên 1 luồng.
//
// HEF: models/best_hailo.hef (YOLO26m, 2 lớp) biên dịch bởi optimization/compile_npu.py: đầu vào uint8 RGB NHWC
//      640x640 (chuẩn hóa trong NPU); 6 đầu ra thô của nhánh one2one - box (H,W,4) khoảng cách ltrb và cls
//      (H,W,nc) logit cho stride 8/16/32. Không NMS: decode + sigmoid + top-k chạy trên CPU (detection_utils.hpp).

#include "detection_utils.hpp"

#include <hailo/hailort.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <sys/mman.h>

#include <array>
#include <atomic>
#include <csignal>
#include <exception>
#include <fstream>
#include <functional>
#include <optional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

namespace {

using Clock = std::chrono::steady_clock;
using edge::BoundedQueue;
using edge::Detection;
using edge::Letterbox;
using edge::OverflowPolicy;

std::atomic<bool> g_stop{false};  // Ctrl+C / SIGTERM / phím 'q' / hết nguồn video / lỗi
std::atomic<int> g_signals{0};

// Lần 1: dừng có trật tự. Lần 2 (vd driver camera treo, app không thoát được): thoát ngay lập tức.
// Chỉ dùng thao tác async-signal-safe (atomic lock-free, std::_Exit).
void on_signal(int) {
    g_stop.store(true);
    if (g_signals.fetch_add(1) >= 1) std::_Exit(130);
}

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// --------------------------------------------------------------------------------------------- //
// Tham số dòng lệnh
// --------------------------------------------------------------------------------------------- //
struct Options {
    std::string hef = "models/best_hailo.hef";
    std::string source = "0";  // số = /dev/videoN (V4L2), "libcamera" = camera CSI Pi, chuỗi có '!' = GStreamer, còn lại = file
    int cam_width = 1280, cam_height = 720, cam_fps = 30;
    double conf = 0.5;  // double: ngưỡng giống hệt benchmark_profiler.py (Python float)
    std::size_t max_det = 300;  // top-k của head end-to-end (Ultralytics max_det)
    bool headless = false;
    std::string save_video;
    std::string stats_csv;
    long max_frames = 0;  // 0 = không giới hạn
    int duration_s = 0;   // 0 = không giới hạn
    bool loop = false;    // phát lại video từ đầu khi hết (đo kiểm dài)
    std::size_t queue_size = 4;
    int drop_mode = -1;  // -1 tự chọn (camera: drop, file: block), 0 block, 1 drop
    double stall_timeout_s = 3.0;  // không có frame mới trong khoảng này -> coi như mất camera / pipeline treo
    std::vector<std::string> names{"traffic_sign", "pedestrian"};
};

[[noreturn]] void usage(const char *prog, int code) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "  --hef PATH          HEF đã biên dịch (mặc định models/best_hailo.hef)\n"
        << "  --source SRC        0 | /dev/video0 | libcamera | video.mp4 | '<gstreamer pipeline> ! appsink'\n"
        << "  --width W --height H --fps F   độ phân giải/FPS yêu cầu từ camera (mặc định 1280x720@30)\n"
        << "  --conf C            ngưỡng tin cậy hiển thị (mặc định 0.5)\n"
        << "  --max-det K         số dự đoán tối đa mỗi frame - top-k của YOLO26 (mặc định 300)\n"
        << "  --headless          không mở cửa sổ (chạy qua SSH / đo kiểm)\n"
        << "  --save out.mp4      ghi video kết quả có overlay\n"
        << "  --stats stats.csv   ghi độ trễ từng frame (cho benchmark_profiler.py)\n"
        << "  --max-frames N      dừng sau N frame;   --duration S  dừng sau S giây;   --loop  lặp video\n"
        << "  --queue N           sức chứa hàng đợi (mặc định 4)\n"
        << "  --drop | --no-drop  bỏ frame cũ ở cả hàng đợi vào lẫn hàng đợi hiển thị (mặc định: camera bỏ, file chờ)\n"
        << "  --stall-timeout S   báo lỗi & thoát nếu không có frame mới trong S giây (mặc định 3; frame đầu 10 s)\n";
    std::exit(code);
}

Options parse_args(int argc, char **argv) {
    Options o;
    auto need = [&](int &i) -> std::string {
        if (i + 1 >= argc) {
            std::cerr << "Thiếu giá trị cho " << argv[i] << "\n";
            usage(argv[0], 2);
        }
        return argv[++i];
    };
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--hef") o.hef = need(i);
            else if (a == "--source") o.source = need(i);
            else if (a == "--width") o.cam_width = std::stoi(need(i));
            else if (a == "--height") o.cam_height = std::stoi(need(i));
            else if (a == "--fps") o.cam_fps = std::stoi(need(i));
            else if (a == "--conf") o.conf = std::stod(need(i));
            else if (a == "--max-det") o.max_det = static_cast<std::size_t>(std::stoul(need(i)));
            else if (a == "--headless") o.headless = true;
            else if (a == "--save") o.save_video = need(i);
            else if (a == "--stats") o.stats_csv = need(i);
            else if (a == "--max-frames") o.max_frames = std::stol(need(i));
            else if (a == "--duration") o.duration_s = std::stoi(need(i));
            else if (a == "--loop") o.loop = true;
            else if (a == "--queue") o.queue_size = static_cast<std::size_t>(std::stoul(need(i)));
            else if (a == "--drop") o.drop_mode = 1;
            else if (a == "--no-drop") o.drop_mode = 0;
            else if (a == "--stall-timeout") o.stall_timeout_s = std::stod(need(i));
            else if (a == "-h" || a == "--help") usage(argv[0], 0);
            else {
                std::cerr << "Tham số không hợp lệ: " << a << "\n";
                usage(argv[0], 2);
            }
        }
    } catch (const std::logic_error &) {  // std::stoi/stof ném invalid_argument / out_of_range
        std::cerr << "Giá trị số không hợp lệ\n";
        usage(argv[0], 2);
    }
    if (o.conf < 0.0 || o.conf > 1.0 || o.max_det == 0 || o.queue_size == 0 || !(o.stall_timeout_s > 0)) {
        std::cerr << "--conf phải trong [0,1], --max-det >= 1, --queue >= 1, --stall-timeout > 0\n";
        usage(argv[0], 2);
    }
    return o;
}

bool is_number(const std::string &s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
}

bool is_live_source(const std::string &src) {
    return is_number(src) || src.rfind("/dev/video", 0) == 0 || src == "libcamera" ||
           src.find('!') != std::string::npos;
}

std::unique_ptr<cv::VideoCapture> open_source(const Options &o) {
    auto cap = std::make_unique<cv::VideoCapture>();
    const std::string &src = o.source;
    if (is_number(src) || src.rfind("/dev/video", 0) == 0) {
        // Camera USB qua V4L2. MJPG giảm băng thông USB, cho phép 720p/1080p ở 30 FPS.
        const int index = is_number(src) ? std::stoi(src) : -1;
        if (index >= 0) cap->open(index, cv::CAP_V4L2);
        else cap->open(src, cv::CAP_V4L2);
        cap->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap->set(cv::CAP_PROP_FRAME_WIDTH, o.cam_width);
        cap->set(cv::CAP_PROP_FRAME_HEIGHT, o.cam_height);
        cap->set(cv::CAP_PROP_FPS, o.cam_fps);
        cap->set(cv::CAP_PROP_BUFFERSIZE, 1);  // giữ ít frame trong driver -> độ trễ thấp
    } else if (src == "libcamera") {
        // Camera CSI của Pi 5 (libcamera) qua GStreamer - cần gói gstreamer1.0-libcamera.
        std::ostringstream p;
        p << "libcamerasrc ! video/x-raw,width=" << o.cam_width << ",height=" << o.cam_height
          << ",framerate=" << o.cam_fps << "/1,format=RGBx ! videoconvert ! video/x-raw,format=BGR"
          << " ! appsink drop=true max-buffers=1 sync=false";
        cap->open(p.str(), cv::CAP_GSTREAMER);
    } else if (src.find('!') != std::string::npos) {
        cap->open(src, cv::CAP_GSTREAMER);
    } else {
        cap->open(src);
    }
    if (!cap->isOpened()) {
        throw std::runtime_error("Không mở được nguồn video: " + src);
    }
    return cap;
}

// --------------------------------------------------------------------------------------------- //
// Pool buffer DMA: cấp phát căn theo trang + map sẵn vào VDevice (tránh map/unmap mỗi frame)
// --------------------------------------------------------------------------------------------- //
std::shared_ptr<std::uint8_t> page_aligned_alloc(std::size_t size) {
    void *addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (addr == MAP_FAILED) throw std::bad_alloc();
    return std::shared_ptr<std::uint8_t>(static_cast<std::uint8_t *>(addr),
                                         [size](std::uint8_t *p) { munmap(p, size); });
}

struct BufferSlot {
    std::shared_ptr<std::uint8_t> input;
    std::vector<std::shared_ptr<std::uint8_t>> outputs;  // cùng thứ tự với danh sách OutputSpec
};

class BufferPool : public std::enable_shared_from_this<BufferPool> {
public:
    BufferPool(hailort::VDevice &vdevice, std::size_t count, std::size_t input_size,
               const std::vector<std::size_t> &output_sizes)
        : free_(count, OverflowPolicy::Block) {
        for (std::size_t i = 0; i < count; ++i) {
            auto slot = std::make_shared<BufferSlot>();
            slot->input = page_aligned_alloc(input_size);
            // Mapping phải bị huỷ TRƯỚC khi giải phóng buffer -> lưu sau slots_ (huỷ theo thứ tự ngược)
            mappings_.push_back(hailort::DmaMappedBuffer::create(vdevice, slot->input.get(), input_size,
                                                                 HAILO_DMA_BUFFER_DIRECTION_H2D)
                                    .expect("Không map được input buffer vào VDevice"));
            for (const std::size_t size : output_sizes) {
                slot->outputs.push_back(page_aligned_alloc(size));
                mappings_.push_back(hailort::DmaMappedBuffer::create(vdevice, slot->outputs.back().get(), size,
                                                                     HAILO_DMA_BUFFER_DIRECTION_D2H)
                                        .expect("Không map được output buffer vào VDevice"));
            }
            slots_.push_back(slot);
            free_.push(slot.get());
        }
    }

    // Lấy 1 slot; trả lại pool tự động khi shared_ptr cuối cùng bị huỷ (kể cả khi có ngoại lệ).
    std::shared_ptr<BufferSlot> acquire(std::chrono::milliseconds timeout) {
        auto raw = free_.pop_for(timeout);
        if (!raw) return nullptr;
        auto self = shared_from_this();
        return std::shared_ptr<BufferSlot>(*raw, [self](BufferSlot *s) { self->free_.push(s); });
    }

private:
    std::vector<std::shared_ptr<BufferSlot>> slots_;
    std::vector<hailort::DmaMappedBuffer> mappings_;
    BoundedQueue<BufferSlot *> free_;
};

// Mô tả một đầu ra thô của HEF YOLO26 (FLOAT32 NHWC sau khi HailoRT giải lượng tử)
struct OutputSpec {
    std::string name;
    std::size_t size = 0;  // byte = H * W * C * 4
    int height = 0, width = 0, channels = 0, stride = 0;
    bool is_box = false;  // true: (H,W,4) ltrb; false: (H,W,nc) logit lớp
};

// Cặp (box, cls) của mỗi stride -> chỉ số trong danh sách OutputSpec, sắp xếp theo stride tăng dần
struct ScaleIndex {
    int stride;
    std::size_t box, cls;
};

// --------------------------------------------------------------------------------------------- //
// Gói dữ liệu đi qua 3 luồng
// --------------------------------------------------------------------------------------------- //
struct FramePacket {
    std::uint64_t id = 0;
    cv::Mat frame;  // ảnh gốc BGR để hiển thị
    std::shared_ptr<BufferSlot> buffers;
    Letterbox lb;
    Clock::time_point t_capture, t_preprocessed, t_infer_start, t_infer_done;
    hailo_status status = HAILO_SUCCESS;
};
using FramePtr = std::shared_ptr<FramePacket>;

// Ghi letterbox + BGR->RGB thẳng vào buffer đầu vào của NPU (không copy thêm).
void preprocess_into(const cv::Mat &bgr, std::uint8_t *dst, int dst_w, int dst_h, Letterbox &lb) {
    lb = edge::compute_letterbox(bgr.cols, bgr.rows, dst_w, dst_h);
    cv::Mat canvas(dst_h, dst_w, CV_8UC3, dst);
    const cv::Scalar pad(114, 114, 114);
    if (lb.pad_top > 0) canvas.rowRange(0, lb.pad_top).setTo(pad);
    if (lb.pad_bottom > 0) canvas.rowRange(dst_h - lb.pad_bottom, dst_h).setTo(pad);
    if (lb.pad_left > 0) canvas.colRange(0, lb.pad_left).setTo(pad);
    if (lb.pad_right > 0) canvas.colRange(dst_w - lb.pad_right, dst_w).setTo(pad);
    cv::Mat roi = canvas(cv::Rect(lb.pad_left, lb.pad_top, lb.new_w, lb.new_h));
    if (lb.new_w == bgr.cols && lb.new_h == bgr.rows) {
        cv::cvtColor(bgr, roi, cv::COLOR_BGR2RGB);
    } else {
        thread_local cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(lb.new_w, lb.new_h), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(resized, roi, cv::COLOR_BGR2RGB);
    }
}

// Lưu lỗi đầu tiên xảy ra ở bất kỳ luồng nào để main ném lại sau khi join.
class ErrorSlot {
public:
    void set(std::exception_ptr e) {
        std::lock_guard<std::mutex> lock(m_);
        if (!error_) error_ = std::move(e);
        g_stop.store(true);
    }
    // Thông điệp lỗi đầu tiên (dùng khi phải buộc thoát, không thể rethrow theo luồng bình thường)
    std::string message() {
        std::lock_guard<std::mutex> lock(m_);
        if (!error_) return {};
        try {
            std::rethrow_exception(error_);
        } catch (const std::exception &e) {
            return e.what();
        } catch (...) {
            return "lỗi không xác định";
        }
    }
    void rethrow_if_any() {
        std::lock_guard<std::mutex> lock(m_);
        if (error_) std::rethrow_exception(error_);
    }

private:
    std::mutex m_;
    std::exception_ptr error_;
};

// --------------------------------------------------------------------------------------------- //
// Luồng 1: Capture + tiền xử lý
// --------------------------------------------------------------------------------------------- //
void capture_loop(const Options &o, cv::VideoCapture &cap, BufferPool &pool, BoundedQueue<FramePtr> &out,
                  int in_w, int in_h, ErrorSlot &errors, std::atomic<std::uint64_t> &captured,
                  std::atomic<bool> &done) {
    try {
        std::uint64_t id = 0;
        const bool live = is_live_source(o.source);
        const auto retry_window = std::chrono::duration<double>(o.stall_timeout_s);
        std::optional<Clock::time_point> failing_since;
        while (!g_stop.load()) {
            if (o.max_frames > 0 && static_cast<long>(id) >= o.max_frames) break;
            auto pkt = std::make_shared<FramePacket>();
            if (!cap.read(pkt->frame) || pkt->frame.empty()) {
                if (!live) {
                    if (o.loop && cap.set(cv::CAP_PROP_POS_FRAMES, 0)) continue;
                    break;  // hết video: kết thúc bình thường
                }
                // Camera chập chờn: thử lại trong cửa sổ --stall-timeout, quá hạn -> lỗi (exit code != 0)
                if (!failing_since) failing_since = Clock::now();
                if (Clock::now() - *failing_since < retry_window) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
                throw std::runtime_error("Mất kết nối camera " + o.source + " (không đọc được frame trong " +
                                         std::to_string(o.stall_timeout_s) + " s)");
            }
            failing_since.reset();
            pkt->t_capture = Clock::now();
            pkt->id = id++;
            // Chờ buffer trống (backpressure khi NPU/hiển thị chậm hơn camera)
            while (!g_stop.load() && !(pkt->buffers = pool.acquire(std::chrono::milliseconds(100)))) {
            }
            if (!pkt->buffers) break;
            preprocess_into(pkt->frame, pkt->buffers->input.get(), in_w, in_h, pkt->lb);
            pkt->t_preprocessed = Clock::now();
            captured.fetch_add(1);
            if (!out.push(std::move(pkt))) break;
        }
    } catch (...) {
        errors.set(std::current_exception());
    }
    out.close();  // báo luồng inference: không còn frame mới
    done.store(true);
}

// Các luồng worker được join trong destructor: ngoại lệ ném ra giữa lúc tạo thread và lúc dừng có trật tự
// không để lại std::thread joinable (std::terminate). stop() đánh thức mọi luồng đang chờ trên hàng đợi.
class ThreadGroup {
public:
    explicit ThreadGroup(std::function<void()> stop) : stop_(std::move(stop)) {}
    ThreadGroup(const ThreadGroup &) = delete;
    ThreadGroup &operator=(const ThreadGroup &) = delete;
    ~ThreadGroup() { stop_and_join(); }

    template <typename... Args>
    void spawn(Args &&...args) {
        threads_.emplace_back(std::forward<Args>(args)...);
    }
    void stop_and_join() {
        if (stop_) {
            stop_();
            stop_ = nullptr;
        }
        for (auto &t : threads_) {
            if (t.joinable()) t.join();
        }
    }

private:
    std::function<void()> stop_;
    std::vector<std::thread> threads_;
};

// --------------------------------------------------------------------------------------------- //
// Luồng 2: Inference bất đồng bộ trên Hailo-8
// --------------------------------------------------------------------------------------------- //
// Đếm job đang chạy trên NPU. Dùng chung qua shared_ptr: callback của HailoRT có thể chạy muộn hơn
// vòng đời của inference_loop, nên không được tham chiếu tới biến cục bộ trên stack.
struct InFlight {
    std::mutex m;
    std::condition_variable cv;
    std::size_t count = 0;
};

void inference_loop(hailort::ConfiguredInferModel &model, BoundedQueue<FramePtr> &in, BoundedQueue<FramePtr> &out,
                    std::size_t in_size, const std::vector<OutputSpec> &outputs, ErrorSlot &errors) {
    auto in_flight = std::make_shared<InFlight>();
    try {
        while (auto pkt = in.pop()) {
            FramePtr frame = std::move(*pkt);
            // Chờ HailoRT nhận thêm request (giới hạn số job song song theo hàng đợi async của model)
            hailo_status status = HAILO_TIMEOUT;
            while (status == HAILO_TIMEOUT && !g_stop.load()) {
                status = model.wait_for_async_ready(std::chrono::milliseconds(1000));
            }
            if (g_stop.load()) break;
            if (status != HAILO_SUCCESS) throw hailort::hailort_error(status, "wait_for_async_ready thất bại");

            auto bindings = model.create_bindings().expect("Không tạo được bindings");
            status = bindings.input()->set_buffer(hailort::MemoryView(frame->buffers->input.get(), in_size));
            if (status != HAILO_SUCCESS) throw hailort::hailort_error(status, "set_buffer(input) thất bại");
            for (std::size_t k = 0; k < outputs.size(); ++k) {
                status = bindings.output(outputs[k].name)
                             ->set_buffer(hailort::MemoryView(frame->buffers->outputs[k].get(), outputs[k].size));
                if (status != HAILO_SUCCESS) {
                    throw hailort::hailort_error(status, "set_buffer(" + outputs[k].name + ") thất bại");
                }
            }

            frame->t_infer_start = Clock::now();
            {
                std::lock_guard<std::mutex> lock(in_flight->m);
                ++in_flight->count;
            }
            // Callback giữ FramePtr (shared_ptr) -> buffer còn sống tới khi NPU ghi xong.
            auto job = model.run_async(bindings, [frame, &out, in_flight](const hailort::AsyncInferCompletionInfo &info) {
                frame->t_infer_done = Clock::now();
                frame->status = info.status;
                out.push(frame);  // hàng đợi kết quả đủ lớn (= số slot) -> không bao giờ chặn
                {
                    std::lock_guard<std::mutex> lock(in_flight->m);
                    --in_flight->count;
                }
                in_flight->cv.notify_all();
            });
            if (!job) {
                {
                    std::lock_guard<std::mutex> lock(in_flight->m);
                    --in_flight->count;
                }
                throw hailort::hailort_error(job.status(), "run_async thất bại");
            }
            job->detach();
        }
    } catch (...) {
        errors.set(std::current_exception());
    }
    // Đợi mọi job đang chạy trên NPU hoàn tất trước khi đóng hàng đợi kết quả
    {
        std::unique_lock<std::mutex> lock(in_flight->m);
        if (!in_flight->cv.wait_for(lock, std::chrono::seconds(5), [&] { return in_flight->count == 0; })) {
            errors.set(std::make_exception_ptr(std::runtime_error("NPU không hoàn tất job trong 5 s")));
        }
    }
    out.close();
}

// --------------------------------------------------------------------------------------------- //
// Luồng 3 (main): hậu xử lý, vẽ, HUD, hiển thị, thống kê
// --------------------------------------------------------------------------------------------- //
const std::array<cv::Scalar, 2> kColors{cv::Scalar(0, 200, 255), cv::Scalar(60, 220, 60)};

void draw_detections(cv::Mat &img, const std::vector<Detection> &dets, const std::vector<std::string> &names) {
    for (const auto &d : dets) {
        const cv::Scalar color = kColors[static_cast<std::size_t>(d.class_id) % kColors.size()];
        const cv::Point p1(static_cast<int>(d.x1), static_cast<int>(d.y1));
        const cv::Point p2(static_cast<int>(d.x2), static_cast<int>(d.y2));
        cv::rectangle(img, p1, p2, color, 2);
        std::ostringstream label;
        label << (d.class_id < static_cast<int>(names.size()) ? names[d.class_id] : std::to_string(d.class_id))
              << ' ' << std::fixed << std::setprecision(2) << d.score;
        int base = 0;
        const cv::Size ts = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &base);
        const int top = std::max(p1.y - ts.height - 6, 0);
        cv::rectangle(img, cv::Point(p1.x, top), cv::Point(p1.x + ts.width + 6, top + ts.height + 6), color, cv::FILLED);
        cv::putText(img, label.str(), cv::Point(p1.x + 3, top + ts.height + 2), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
}

void draw_hud(cv::Mat &img, const std::vector<std::string> &lines) {
    const int line_h = 24;
    cv::Mat roi = img(cv::Rect(0, 0, std::min(img.cols, 430), std::min(img.rows, line_h * static_cast<int>(lines.size()) + 10)));
    cv::Mat overlay(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
    cv::addWeighted(overlay, 0.55, roi, 0.45, 0, roi);  // nền tối bán trong suốt
    for (std::size_t i = 0; i < lines.size(); ++i) {
        cv::putText(img, lines[i], cv::Point(8, 22 + static_cast<int>(i) * line_h), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

double read_soc_temp() {
    std::ifstream f("/sys/class/thermal/thermal_zone0/temp");
    long milli = 0;
    return (f >> milli) ? milli / 1000.0 : -1.0;
}

}  // namespace

int main(int argc, char **argv) {
    const Options opt = parse_args(argc, argv);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    try {
        // ---- Khởi tạo HailoRT ----
        auto vdevice = hailort::VDevice::create().expect("Không tạo được VDevice (kiểm tra Hailo-8 và driver)");
        auto infer_model = vdevice->create_infer_model(opt.hef).expect("Không nạp được HEF: " + opt.hef);
        infer_model->set_batch_size(1);

        auto input = infer_model->input().expect("HEF phải có đúng 1 đầu vào");
        input.set_format_type(HAILO_FORMAT_TYPE_UINT8);  // ảnh uint8 RGB, chia 255 nằm trong NPU

        // 6 đầu ra thô YOLO26: HailoRT giải lượng tử (a16 -> float) và trả layout NHWC
        std::vector<OutputSpec> outputs;
        for (const auto &name : infer_model->get_output_names()) {
            auto stream = infer_model->output(name).expect("Không truy cập được đầu ra " + name);
            stream.set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
            stream.set_format_order(HAILO_FORMAT_ORDER_NHWC);
            const hailo_3d_image_shape_t shp = stream.shape();
            OutputSpec spec;
            spec.name = name;
            spec.height = static_cast<int>(shp.height);
            spec.width = static_cast<int>(shp.width);
            spec.channels = static_cast<int>(shp.features);
            spec.is_box = spec.channels == 4;
            outputs.push_back(spec);
        }

        const hailo_3d_image_shape_t in_shape = input.shape();
        const int in_h = static_cast<int>(in_shape.height), in_w = static_cast<int>(in_shape.width);
        if (in_shape.features != 3) throw std::runtime_error("Đầu vào phải có 3 kênh RGB");
        const std::size_t in_size = input.get_frame_size();
        if (in_size != static_cast<std::size_t>(in_w) * in_h * 3) {
            throw std::runtime_error("Đầu vào HEF không phải uint8 NHWC " + std::to_string(in_w) + "x" +
                                     std::to_string(in_h) + "x3 (frame " + std::to_string(in_size) + " B)");
        }
        // Ghép (box, cls) theo stride = cạnh đầu vào / chiều cao feature map
        if (outputs.size() != 6) {
            throw std::runtime_error("HEF có " + std::to_string(outputs.size()) +
                                     " đầu ra; YOLO26 NMS-free cần 6 (box + cls x 3 stride) - xuất lại bằng "
                                     "optimization/export_onnx.py rồi compile_npu.py");
        }
        int num_classes = -1;
        std::vector<ScaleIndex> scales_idx;
        for (std::size_t k = 0; k < outputs.size(); ++k) {
            OutputSpec &o = outputs[k];
            if (o.height <= 0 || in_h % o.height != 0) throw std::runtime_error("Shape đầu ra lạ: " + o.name);
            o.stride = in_h / o.height;
            if (!o.is_box) {
                if (num_classes != -1 && num_classes != o.channels) {
                    throw std::runtime_error("Các đầu ra lớp có số kênh khác nhau");
                }
                num_classes = o.channels;
            }
        }
        for (std::size_t k = 0; k < outputs.size(); ++k) {
            if (!outputs[k].is_box) continue;
            for (std::size_t j = 0; j < outputs.size(); ++j) {
                if (!outputs[j].is_box && outputs[j].stride == outputs[k].stride) {
                    scales_idx.push_back({outputs[k].stride, k, j});
                }
            }
        }
        std::sort(scales_idx.begin(), scales_idx.end(),
                  [](const ScaleIndex &a, const ScaleIndex &b) { return a.stride < b.stride; });
        if (scales_idx.size() != 3 || num_classes == 4) {
            throw std::runtime_error("Không ghép được 3 cặp (box 4 kênh, cls nc kênh) theo stride từ đầu ra HEF");
        }
        if (static_cast<std::size_t>(num_classes) != opt.names.size()) {
            std::cerr << "[CẢNH BÁO] HEF có " << num_classes << " lớp, app khai báo " << opt.names.size()
                      << " tên lớp\n";
        }

        auto configured = infer_model->configure().expect("Không cấu hình được model trên NPU");
        std::vector<std::size_t> output_sizes;
        for (OutputSpec &o : outputs) {  // đọc sau configure (đã áp dụng FLOAT32 + NHWC)
            o.size = infer_model->output(o.name).expect("output " + o.name).get_frame_size();
            const std::size_t expect_bytes = static_cast<std::size_t>(o.height) * o.width * o.channels * sizeof(float);
            if (o.size != expect_bytes) {
                throw std::runtime_error("Đầu ra " + o.name + " có " + std::to_string(o.size) + " B, kỳ vọng " +
                                         std::to_string(expect_bytes) + " B (FLOAT32 NHWC)");
            }
            output_sizes.push_back(o.size);
        }
        const std::size_t async_depth = configured.get_async_queue_size().expect("get_async_queue_size thất bại");

        // Đọc nhiệt độ NPU qua thiết bị vật lý (không bắt buộc - một số HailoRT/driver không hỗ trợ)
        hailort::Device *npu = nullptr;
        if (auto devs = vdevice->get_physical_devices(); devs && !devs->empty()) npu = &devs->front().get();

        std::cout << "HEF " << opt.hef << " | input " << in_w << "x" << in_h << "x3 uint8 (" << in_size
                  << " B) | YOLO26 end-to-end " << num_classes << " lớp, top-" << opt.max_det
                  << " | async queue " << async_depth << std::endl;
        for (const ScaleIndex &si : scales_idx) {
            const OutputSpec &b = outputs[si.box], &c = outputs[si.cls];
            std::cout << "  stride " << si.stride << ": box " << b.name << " " << b.height << "x" << b.width << "x"
                      << b.channels << " | cls " << c.name << " " << c.height << "x" << c.width << "x" << c.channels
                      << std::endl;
        }

        // ---- Nguồn video ----
        auto cap = open_source(opt);
        const bool live = is_live_source(opt.source);
        const OverflowPolicy policy =
            opt.drop_mode == 1 || (opt.drop_mode == -1 && live) ? OverflowPolicy::DropOldest : OverflowPolicy::Block;
        std::cout << "Nguồn " << opt.source << " " << cap->get(cv::CAP_PROP_FRAME_WIDTH) << "x"
                  << cap->get(cv::CAP_PROP_FRAME_HEIGHT) << " | hàng đợi " << opt.queue_size << " ("
                  << (policy == OverflowPolicy::DropOldest ? "bỏ frame cũ" : "chờ") << ")" << std::endl;

        // ---- Hàng đợi & pool buffer ----
        // Số slot = frame trong hàng đợi vào + job async trên NPU + 2 frame đang tiền xử lý/hiển thị
        const std::size_t slots = opt.queue_size + async_depth + 2;
        auto pool = std::make_shared<BufferPool>(*vdevice, slots, in_size, output_sizes);
        BoundedQueue<FramePtr> pre_queue(opt.queue_size, policy);
        // Hàng đợi kết quả không bao giờ đầy (sức chứa = số slot); với nguồn trực tiếp, luồng hiển thị tự bỏ
        // các frame cũ đang chờ và chỉ vẽ frame mới nhất (xem vòng lặp chính) -> độ trễ không dồn ứ.
        BoundedQueue<FramePtr> post_queue(slots, OverflowPolicy::Block);
        const bool drop_late = policy == OverflowPolicy::DropOldest;
        ErrorSlot errors;
        std::atomic<std::uint64_t> captured{0};
        std::atomic<bool> capture_done{false};

        // Mở file thống kê & đọc thuộc tính nguồn TRƯỚC khi tạo luồng: ngoại lệ ở đây không để lại thread
        // joinable (std::terminate), và cv::VideoCapture không được truy cập đồng thời từ 2 luồng.
        std::unique_ptr<cv::VideoWriter> writer;
        std::unique_ptr<std::ofstream> csv;
        if (!opt.stats_csv.empty()) {
            csv = std::make_unique<std::ofstream>(opt.stats_csv);
            if (!*csv) throw std::runtime_error("Không ghi được " + opt.stats_csv);
            *csv << "frame_id,t_ms,preprocess_ms,queue_ms,npu_ms,postprocess_ms,e2e_ms,detections,soc_temp_c,npu_temp_c\n";
        }
        const double video_fps = live ? opt.cam_fps : std::max(1.0, cap->get(cv::CAP_PROP_FPS));

        // Khai báo SAU các hàng đợi -> bị huỷ (join) TRƯỚC chúng
        ThreadGroup workers([&] {
            g_stop.store(true);
            pre_queue.close();
            post_queue.close();
        });
        workers.spawn(capture_loop, std::cref(opt), std::ref(*cap), std::ref(*pool), std::ref(pre_queue), in_w, in_h,
                      std::ref(errors), std::ref(captured), std::ref(capture_done));
        workers.spawn(inference_loop, std::ref(configured), std::ref(pre_queue), std::ref(post_queue), in_size,
                      std::cref(outputs), std::ref(errors));

        // ---- Luồng 3: hậu xử lý & hiển thị ----
        edge::LatencyStats st_pre, st_queue, st_npu, st_post, st_e2e;
        edge::FpsMeter fps_meter;
        const auto t_start = Clock::now();
        auto t_last_sensor = t_start - std::chrono::seconds(2);
        double soc_temp = -1, npu_temp = -1;
        std::uint64_t shown = 0, late_drops = 0;
        auto t_last_frame = t_start;
        bool got_frame = false;
        const auto stall_limit = std::chrono::duration<double>(opt.stall_timeout_s);
        const auto first_frame_limit = std::chrono::duration<double>(std::max(10.0, opt.stall_timeout_s));

        try {
            while (!g_stop.load()) {
                // Chờ có timeout: GUI, phím 'q', --duration và watchdog vẫn chạy khi không có frame
                auto item = post_queue.pop_for(std::chrono::milliseconds(50));
                const auto t_wait_end = Clock::now();
                if (!item) {
                    if (post_queue.closed() && post_queue.size() == 0) break;  // nguồn đã hết, pipeline xả xong
                    if (!opt.headless) {
                        const int key = cv::waitKey(1) & 0xFF;
                        if (key == 'q' || key == 27) g_stop.store(true);
                    }
                    if (opt.duration_s > 0 && t_wait_end - t_start >= std::chrono::seconds(opt.duration_s)) {
                        g_stop.store(true);
                    }
                    if (t_wait_end - t_last_frame > (got_frame ? stall_limit : first_frame_limit)) {
                        throw std::runtime_error(
                            "Không nhận được frame nào trong " +
                            std::to_string(std::chrono::duration<double>(t_wait_end - t_last_frame).count()) +
                            " s - camera mất kết nối hoặc pipeline bị treo");
                    }
                    continue;
                }
                FramePtr f = std::move(*item);
                if (f->status != HAILO_SUCCESS) throw hailort::hailort_error(f->status, "Job suy luận thất bại");
                if (drop_late) {  // nguồn trực tiếp: bỏ các kết quả cũ đang chờ, chỉ hiển thị frame mới nhất
                    while (auto newer = post_queue.pop_for(std::chrono::milliseconds(0))) {
                        if ((*newer)->status != HAILO_SUCCESS) {
                            throw hailort::hailort_error((*newer)->status, "Job suy luận thất bại");
                        }
                        f = std::move(*newer);  // frame cũ bị huỷ -> buffer DMA về pool ngay
                        ++late_drops;
                    }
                }
                got_frame = true;
                t_last_frame = t_wait_end;
                const auto t_post_start = Clock::now();
                std::vector<edge::ScaleOutputs> views;
                views.reserve(scales_idx.size());
                for (const ScaleIndex &si : scales_idx) {
                    const OutputSpec &b = outputs[si.box], &c = outputs[si.cls];
                    views.push_back({si.stride,
                                     {reinterpret_cast<const float *>(f->buffers->outputs[si.box].get()), b.height,
                                      b.width, b.channels, b.size},
                                     {reinterpret_cast<const float *>(f->buffers->outputs[si.cls].get()), c.height,
                                      c.width, c.channels, c.size}});
                }
                auto dets = edge::decode_end2end(views, num_classes, opt.conf, opt.max_det, f->lb);
                f->buffers.reset();  // trả buffer DMA về pool sớm nhất có thể

                draw_detections(f->frame, dets, opt.names);
                const auto t_now = Clock::now();
                if (t_now - t_last_sensor >= std::chrono::seconds(1)) {  // cảm biến: 1 Hz là đủ
                    t_last_sensor = t_now;
                    soc_temp = read_soc_temp();
                    if (npu) {
                        if (auto t = npu->get_chip_temperature()) npu_temp = t->ts0_temperature;
                    }
                }
                const double pre = ms_between(f->t_capture, f->t_preprocessed);
                const double queue = ms_between(f->t_preprocessed, f->t_infer_start);
                const double npu_ms = ms_between(f->t_infer_start, f->t_infer_done);
                const double fps = fps_meter.tick(t_now);
                const double post = ms_between(t_post_start, t_now);
                const double e2e = ms_between(f->t_capture, t_now);
                st_pre.add(pre), st_queue.add(queue), st_npu.add(npu_ms), st_post.add(post), st_e2e.add(e2e);

                std::ostringstream l1, l2, l3, l4;
                l1 << std::fixed << std::setprecision(1) << "FPS: " << fps << "   Objects: " << dets.size();
                l2 << std::fixed << std::setprecision(1) << "E2E latency: " << e2e << " ms (avg " << st_e2e.ema() << ")";
                l3 << std::fixed << std::setprecision(1) << "NPU: " << npu_ms << " ms  Pre: " << pre << "  Post: " << post;
                l4 << std::fixed << std::setprecision(1) << "SoC " << soc_temp << " C  NPU " << npu_temp
                   << " C  drop " << pre_queue.dropped() + late_drops;
                draw_hud(f->frame, {l1.str(), l2.str(), l3.str(), l4.str()});

                if (!opt.save_video.empty()) {
                    if (!writer) {
                        writer = std::make_unique<cv::VideoWriter>(opt.save_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                                                   video_fps, f->frame.size());
                        if (!writer->isOpened()) throw std::runtime_error("Không mở được file ghi " + opt.save_video);
                    }
                    writer->write(f->frame);
                }
                if (csv) {
                    *csv << f->id << ',' << std::fixed << std::setprecision(3) << ms_between(t_start, f->t_capture) << ','
                         << pre << ',' << queue << ',' << npu_ms << ',' << post << ',' << e2e << ',' << dets.size()
                         << ',' << soc_temp << ',' << npu_temp << '\n';
                }
                if (!opt.headless) {
                    cv::imshow("BKAuto - YOLO26m @ Hailo-8", f->frame);
                    const int key = cv::waitKey(1) & 0xFF;
                    if (key == 'q' || key == 27) g_stop.store(true);
                }
                ++shown;
                if (opt.duration_s > 0 && t_now - t_start >= std::chrono::seconds(opt.duration_s)) g_stop.store(true);
            }
        } catch (...) {
            errors.set(std::current_exception());
        }

        // ---- Dừng có trật tự: đóng hàng đợi -> join -> giải phóng theo thứ tự ngược ----
        g_stop.store(true);
        pre_queue.close();
        post_queue.close();
        while (post_queue.pop_for(std::chrono::milliseconds(0))) {
        }  // trả buffer của frame chưa hiển thị
        // Luồng capture có thể kẹt trong cap.read() khi driver camera / GStreamer treo: chờ tối đa 3 s
        const auto join_deadline = Clock::now() + std::chrono::seconds(3);
        while (!capture_done.load() && Clock::now() < join_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (!capture_done.load()) {
            if (const std::string msg = errors.message(); !msg.empty()) std::cerr << "[LỖI] " << msg << std::endl;
            std::cerr << "[LỖI] Luồng camera không phản hồi (cap.read() bị treo trong driver) - dừng NPU và buộc thoát"
                      << std::endl;
            configured.shutdown();
            if (csv) csv->flush();
            std::cout.flush();
            std::_Exit(3);  // không thể join luồng đang kẹt trong driver; kernel giải phóng /dev/hailo0 khi thoát
        }
        workers.stop_and_join();
        while (post_queue.pop_for(std::chrono::milliseconds(0))) {
        }
        // Dừng hẳn pipeline HailoRT trước khi huỷ pool buffer / hàng đợi mà callback có thể chạm tới
        if (const auto st = configured.shutdown(); st != HAILO_SUCCESS) {
            std::cerr << "[CẢNH BÁO] ConfiguredInferModel::shutdown status=" << st << std::endl;
        }
        if (!opt.headless) cv::destroyAllWindows();
        errors.rethrow_if_any();

        const double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
        std::cout << std::fixed << std::setprecision(2) << "\n==== Tổng kết (" << shown << " frame hiển thị / "
                  << captured.load() << " frame đọc, bỏ " << pre_queue.dropped() << " ở hàng đợi vào + " << late_drops
                  << " ở hàng đợi hiển thị, " << elapsed << " s) ====\n"
                  << "Throughput        : " << (elapsed > 0 ? shown / elapsed : 0.0) << " FPS\n";
        auto row = [](const char *name, const edge::LatencyStats &s) {
            std::cout << std::left << std::setw(18) << name << ": mean " << s.mean() << " | p50 " << s.percentile(50)
                      << " | p95 " << s.percentile(95) << " | p99 " << s.percentile(99) << " ms\n";
        };
        row("Preprocess", st_pre), row("Queue wait", st_queue), row("NPU (async job)", st_npu);
        row("Postprocess+draw", st_post), row("End-to-End", st_e2e);
        return 0;
    } catch (const hailort::hailort_error &e) {
        std::cerr << "[LỖI HailoRT] status=" << e.status() << ": " << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "[LỖI] " << e.what() << std::endl;
        return 1;
    }
}
