// detection_utils.hpp - Thành phần thuần C++17 (không phụ thuộc HailoRT/OpenCV) dùng chung cho
// inference_app.cpp và unit test: hàng đợi thread-safe, hậu xử lý YOLO26 end-to-end (NMS-free),
// hình học letterbox và thống kê độ trễ.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace edge {

// --------------------------------------------------------------------------------------------- //
// Hàng đợi Producer-Consumer có giới hạn
// --------------------------------------------------------------------------------------------- //
enum class OverflowPolicy {
    Block,      // video file / đo kiểm: không bỏ frame nào, producer chờ consumer
    DropOldest  // camera thời gian thực: bỏ frame cũ nhất để luôn xử lý frame mới nhất (độ trễ thấp)
};

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity, OverflowPolicy policy = OverflowPolicy::Block)
        : capacity_(std::max<std::size_t>(1, capacity)), policy_(policy) {}

    BoundedQueue(const BoundedQueue &) = delete;
    BoundedQueue &operator=(const BoundedQueue &) = delete;

    // Trả về false nếu hàng đợi đã đóng (phần tử bị bỏ).
    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (policy_ == OverflowPolicy::Block) {
            not_full_.wait(lock, [this] { return closed_ || items_.size() < capacity_; });
        } else if (!closed_ && items_.size() >= capacity_) {
            items_.pop_front();
            ++dropped_;
        }
        if (closed_) {
            return false;
        }
        items_.push_back(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    // Chờ tới khi có phần tử; trả về std::nullopt khi hàng đợi đã đóng VÀ rỗng.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return closed_ || !items_.empty(); });
        return take_locked();
    }

    // Như pop() nhưng có timeout; std::nullopt nếu hết giờ hoặc đã đóng và rỗng.
    template <typename Rep, typename Period>
    std::optional<T> pop_for(const std::chrono::duration<Rep, Period> &timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait_for(lock, timeout, [this] { return closed_ || !items_.empty(); });
        return take_locked();
    }

    // Đóng hàng đợi: đánh thức mọi luồng đang chờ; consumer vẫn lấy nốt phần tử còn lại.
    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    bool closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }
    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return items_.size();
    }
    std::uint64_t dropped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }

private:
    std::optional<T> take_locked() {
        if (items_.empty()) {
            return std::nullopt;
        }
        T item = std::move(items_.front());
        items_.pop_front();
        not_full_.notify_one();
        return item;
    }

    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<T> items_;
    const std::size_t capacity_;
    const OverflowPolicy policy_;
    bool closed_ = false;
    std::uint64_t dropped_ = 0;
};

// --------------------------------------------------------------------------------------------- //
// Letterbox: khớp ultralytics LetterBox(auto=False, center=True) và calibrate_ptq.py
// --------------------------------------------------------------------------------------------- //
struct Letterbox {
    float scale = 1.f;         // tỉ lệ resize ảnh gốc -> ảnh trong khung model
    int new_w = 0, new_h = 0;  // kích thước ảnh sau resize (chưa pad)
    int pad_left = 0, pad_top = 0, pad_right = 0, pad_bottom = 0;
    int src_w = 0, src_h = 0;
    int dst_w = 0, dst_h = 0;
};

inline int py_round(double v) {
    // Python round(): làm tròn nửa về số chẵn - giữ đúng từng pixel như ultralytics/calibrate_ptq.py
    const double r = std::nearbyint(v);  // chế độ làm tròn mặc định FE_TONEAREST = banker's rounding
    return static_cast<int>(r);
}

inline Letterbox compute_letterbox(int src_w, int src_h, int dst_w, int dst_h) {
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        throw std::invalid_argument("compute_letterbox: kích thước không hợp lệ");
    }
    Letterbox lb;
    lb.src_w = src_w;
    lb.src_h = src_h;
    lb.dst_w = dst_w;
    lb.dst_h = dst_h;
    const double r = std::min(static_cast<double>(dst_h) / src_h, static_cast<double>(dst_w) / src_w);
    lb.scale = static_cast<float>(r);
    lb.new_w = py_round(src_w * r);
    lb.new_h = py_round(src_h * r);
    const double dw = (dst_w - lb.new_w) / 2.0;
    const double dh = (dst_h - lb.new_h) / 2.0;
    lb.pad_top = py_round(dh - 0.1);
    lb.pad_bottom = py_round(dh + 0.1);
    lb.pad_left = py_round(dw - 0.1);
    lb.pad_right = py_round(dw + 0.1);
    return lb;
}

// --------------------------------------------------------------------------------------------- //
// Kết quả phát hiện
// --------------------------------------------------------------------------------------------- //
struct Detection {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;  // pixel (khung model hoặc ảnh gốc - xem hàm trả về)
    float score = 0;
    int class_id = 0;
};

// Đổi box xyxy (pixel trên khung model đã letterbox) -> pixel trên ảnh gốc, cắt về biên ảnh.
inline Detection unletterbox_px(float x1, float y1, float x2, float y2, float score, int class_id,
                                const Letterbox &lb) {
    auto map_x = [&](float v) { return std::clamp((v - lb.pad_left) / lb.scale, 0.f, static_cast<float>(lb.src_w)); };
    auto map_y = [&](float v) { return std::clamp((v - lb.pad_top) / lb.scale, 0.f, static_cast<float>(lb.src_h)); };
    return Detection{map_x(x1), map_y(y1), map_x(x2), map_y(y2), score, class_id};
}

// --------------------------------------------------------------------------------------------- //
// Hậu xử lý YOLO26 end-to-end (NMS-free) trên đầu ra thô của HEF
// --------------------------------------------------------------------------------------------- //
// Mỗi stride (8/16/32) có 2 tensor FLOAT32 layout NHWC (HailoRT đã giải lượng tử):
//   box : (H, W, 4)  = khoảng cách (l, t, r, b) tính theo đơn vị stride (reg_max = 1, KHÔNG có DFL)
//   cls : (H, W, nc) = logit lớp (sigmoid chưa áp dụng - không đặt sigmoid on-chip)
//
// Thuật toán - GIỐNG HỆT decode_end2end() trong edge_deployment/benchmark_profiler.py (cùng thứ tự, cùng
// lớp, box và score trùng từng bit):
//   1. Ngưỡng: score > th  <=>  logit > float32(log(th / (1 - th)))  (tính bằng double rồi làm tròn về float32;
//      th <= 0 -> mọi ô, th >= 1 -> không ô nào). So sánh trên float32.
//   2. Ứng viên = mọi cặp (anchor, lớp) vượt ngưỡng, liệt kê theo thứ tự (stride theo danh sách scales, ô
//      lưới theo hàng y rồi cột x, lớp) - chính là thứ tự np.nonzero trên ma trận (anchor, lớp) của Python.
//   3. Sắp xếp giảm dần theo LOGIT (sigmoid đơn điệu -> cùng thứ hạng với score, nhưng logit là đầu vào float32
//      giống hệt nhau ở 2 ngôn ngữ, không phụ thuộc exp()); bằng nhau thì giữ thứ tự ở bước 2 (= argsort stable).
//      Giữ max_det phần tử đầu.
//   4. Với phần tử được chọn: anchor = (x + 0.5, y + 0.5) (float32);
//      box xyxy = ((ax - l) * s, (ay - t) * s, (ax + r) * s, (ay + b) * s) (float32);
//      score = float32(1 / (1 + exp(-logit))) tính bằng double.
// Tương đương ultralytics Detect._inference + Detect.postprocess (top-k 2 tầng): mọi cặp thuộc top-k toàn cục có
// anchor nằm trong top-k anchor theo điểm max, vì số anchor có điểm max lớn hơn nó luôn < k. Không có NMS:
// nhánh one2one được huấn luyện để mỗi vật thể chỉ sinh một dự đoán.
struct TensorView {
    const float *data = nullptr;
    int height = 0, width = 0, channels = 0;
    std::size_t size_bytes = 0;  // kích thước buffer thật - để kiểm tra biên
};

struct ScaleOutputs {
    int stride = 0;
    TensorView box;  // (H, W, 4)
    TensorView cls;  // (H, W, nc)
};

// Tính bằng double rồi làm tròn về float32 (như benchmark_profiler.py): kết quả trùng từng bit giữa 2 ngôn ngữ,
// không phụ thuộc cài đặt expf() float32 của libm / numpy (vốn có thể lệch 1-2 ulp).
inline float sigmoid(float x) { return static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(x)))); }

// Ngưỡng trên logit tương đương score > score_threshold (float32, giống benchmark_profiler.py)
inline float logit_threshold(double score_threshold) {
    if (score_threshold <= 0.0) return -std::numeric_limits<float>::infinity();
    if (score_threshold >= 1.0) return std::numeric_limits<float>::infinity();
    return static_cast<float>(std::log(score_threshold / (1.0 - score_threshold)));
}

// Giải mã ra toạ độ pixel trên khung model (640x640 đã letterbox), chưa đổi về ảnh gốc.
inline std::vector<Detection> decode_end2end_model(const std::vector<ScaleOutputs> &scales, int num_classes,
                                                   double score_threshold, std::size_t max_det) {
    struct Candidate {
        float logit;
        int scale_idx;
        int cell;
        int class_id;
    };
    if (num_classes <= 0) throw std::invalid_argument("decode_end2end: num_classes phải > 0");

    // ---- Bước 1-2: lọc theo ngưỡng logit (bỏ qua exp() cho phần lớn ô lưới) ----
    const float logit_th = logit_threshold(score_threshold);
    std::vector<Candidate> cands;
    for (std::size_t s = 0; s < scales.size(); ++s) {
        const ScaleOutputs &so = scales[s];
        const TensorView &box = so.box, &cls = so.cls;
        if (box.data == nullptr || cls.data == nullptr) throw std::runtime_error("decode_end2end: tensor rỗng");
        if (box.channels != 4 || cls.channels != num_classes || box.height != cls.height ||
            box.width != cls.width || so.stride <= 0) {
            throw std::runtime_error("decode_end2end: shape đầu ra không khớp ở stride " + std::to_string(so.stride));
        }
        const std::size_t cells = static_cast<std::size_t>(cls.height) * static_cast<std::size_t>(cls.width);
        if (box.size_bytes < cells * 4 * sizeof(float) ||
            cls.size_bytes < cells * static_cast<std::size_t>(num_classes) * sizeof(float)) {
            throw std::runtime_error("decode_end2end: buffer nhỏ hơn shape khai báo ở stride " +
                                     std::to_string(so.stride));
        }
        for (std::size_t cell = 0; cell < cells; ++cell) {
            const float *logits = cls.data + cell * static_cast<std::size_t>(num_classes);
            for (int c = 0; c < num_classes; ++c) {
                if (logits[c] > logit_th) {  // NaN luôn bị loại (so sánh với NaN = false), như numpy
                    cands.push_back({logits[c], static_cast<int>(s), static_cast<int>(cell), c});
                }
            }
        }
    }

    // ---- Bước 3: top-k theo logit giảm dần; bằng nhau -> thứ tự liệt kê (thứ tự toàn phần, xác định) ----
    auto before = [](const Candidate &a, const Candidate &b) {
        if (a.logit != b.logit) return a.logit > b.logit;
        return std::tie(a.scale_idx, a.cell, a.class_id) < std::tie(b.scale_idx, b.cell, b.class_id);
    };
    if (cands.size() > max_det) {
        std::partial_sort(cands.begin(), cands.begin() + static_cast<std::ptrdiff_t>(max_det), cands.end(), before);
        cands.resize(max_det);
    } else {
        std::sort(cands.begin(), cands.end(), before);
    }

    // ---- Bước 4: chỉ decode box + sigmoid cho các phần tử được giữ ----
    std::vector<Detection> dets;
    dets.reserve(cands.size());
    for (const Candidate &k : cands) {
        const ScaleOutputs &so = scales[static_cast<std::size_t>(k.scale_idx)];
        const int w = so.box.width;
        const float ax = static_cast<float>(k.cell % w) + 0.5f;
        const float ay = static_cast<float>(k.cell / w) + 0.5f;
        const float *ltrb = so.box.data + static_cast<std::size_t>(k.cell) * 4;
        const auto st = static_cast<float>(so.stride);
        dets.push_back(Detection{(ax - ltrb[0]) * st, (ay - ltrb[1]) * st, (ax + ltrb[2]) * st, (ay + ltrb[3]) * st,
                                 sigmoid(k.logit), k.class_id});
    }
    return dets;
}

// Giải mã + đổi về toạ độ pixel trên ảnh gốc (bỏ letterbox, cắt về biên ảnh) - dùng trong inference_app.
inline std::vector<Detection> decode_end2end(const std::vector<ScaleOutputs> &scales, int num_classes,
                                             double score_threshold, std::size_t max_det, const Letterbox &lb) {
    std::vector<Detection> dets = decode_end2end_model(scales, num_classes, score_threshold, max_det);
    for (Detection &d : dets) d = unletterbox_px(d.x1, d.y1, d.x2, d.y2, d.score, d.class_id, lb);
    return dets;
}

// --------------------------------------------------------------------------------------------- //
// Thống kê độ trễ: trung bình trượt cho HUD + tổng hợp trung bình/p50/p95/p99 khi kết thúc
// --------------------------------------------------------------------------------------------- //
class LatencyStats {
public:
    // Bộ nhớ có giới hạn khi chạy liên tục nhiều giờ: trung bình tính trên MỌI mẫu, phân vị tính trên
    // mẫu đại diện (reservoir sampling, thuật toán R) tối đa `capacity` phần tử.
    explicit LatencyStats(std::size_t capacity = 100000) : capacity_(std::max<std::size_t>(1, capacity)) {}

    void add(double value_ms) {
        ++count_;
        sum_ += value_ms;
        ema_ = count_ == 1 ? value_ms : 0.9 * ema_ + 0.1 * value_ms;
        if (reservoir_.size() < capacity_) {
            reservoir_.push_back(value_ms);
        } else {
            std::uniform_int_distribution<std::uint64_t> pick(0, count_ - 1);
            const std::uint64_t j = pick(rng_);
            if (j < capacity_) reservoir_[static_cast<std::size_t>(j)] = value_ms;
        }
    }
    double ema() const { return ema_; }
    std::uint64_t count() const { return count_; }
    std::size_t stored() const { return reservoir_.size(); }
    double mean() const { return count_ ? sum_ / static_cast<double>(count_) : 0.0; }
    double percentile(double p) const {
        if (reservoir_.empty()) return 0.0;
        std::vector<double> sorted(reservoir_);
        std::sort(sorted.begin(), sorted.end());
        const double pos = std::clamp(p, 0.0, 100.0) / 100.0 * static_cast<double>(sorted.size() - 1);
        const auto lo = static_cast<std::size_t>(std::floor(pos));
        const auto hi = static_cast<std::size_t>(std::ceil(pos));
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - static_cast<double>(lo));
    }

private:
    std::size_t capacity_;
    std::vector<double> reservoir_;
    std::uint64_t count_ = 0;
    double sum_ = 0.0;
    double ema_ = 0.0;
    std::mt19937_64 rng_{0x9E3779B97F4A7C15ULL};  // seed cố định: kết quả tái lập được
};

// FPS tức thời theo cửa sổ trượt 1 giây.
class FpsMeter {
public:
    using Clock = std::chrono::steady_clock;
    double tick(Clock::time_point now = Clock::now()) {
        stamps_.push_back(now);
        while (!stamps_.empty() && now - stamps_.front() > std::chrono::seconds(1)) {
            stamps_.pop_front();
        }
        if (stamps_.size() < 2) return 0.0;
        const double span = std::chrono::duration<double>(stamps_.back() - stamps_.front()).count();
        return span > 0 ? static_cast<double>(stamps_.size() - 1) / span : 0.0;
    }

private:
    std::deque<Clock::time_point> stamps_;
};

}  // namespace edge
