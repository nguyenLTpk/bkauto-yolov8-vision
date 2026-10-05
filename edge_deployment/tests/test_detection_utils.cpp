// Unit test cho detection_utils.hpp (không cần HailoRT/OpenCV):
//   cmake -B build -DBKAUTO_BUILD_TESTS=ON && cmake --build build && ctest --test-dir build
#include "detection_utils.hpp"

#include <array>
#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

namespace {
int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);          \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

bool near(float a, float b, float eps = 1e-3f) { return std::fabs(a - b) <= eps; }

void test_letterbox_matches_ultralytics() {
    // {src_w, src_h, new_w, new_h, pad_left, pad_top, pad_right, pad_bottom} sinh từ
    // ultralytics.data.augment.LetterBox((640, 640), auto=False, center=True)
    const int cases[][8] = {
        {1280, 720, 640, 360, 0, 140, 0, 140}, {640, 480, 640, 480, 0, 80, 0, 80},
        {1920, 1080, 640, 360, 0, 140, 0, 140}, {721, 405, 640, 360, 0, 140, 0, 140},
        {333, 777, 274, 640, 183, 0, 183, 0},   {640, 640, 640, 640, 0, 0, 0, 0},
        {1000, 3, 640, 2, 0, 319, 0, 319},      {1366, 768, 640, 360, 0, 140, 0, 140},
        {2592, 1944, 640, 480, 0, 80, 0, 80},
    };
    for (const auto &c : cases) {
        const auto lb = edge::compute_letterbox(c[0], c[1], 640, 640);
        CHECK(lb.new_w == c[2] && lb.new_h == c[3]);
        CHECK(lb.pad_left == c[4] && lb.pad_top == c[5] && lb.pad_right == c[6] && lb.pad_bottom == c[7]);
        CHECK(lb.pad_left + lb.new_w + lb.pad_right == 640 && lb.pad_top + lb.new_h + lb.pad_bottom == 640);
    }
    bool threw = false;
    try {
        edge::compute_letterbox(0, 10, 640, 640);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

void test_unletterbox_roundtrip() {
    const auto lb = edge::compute_letterbox(1280, 720, 640, 640);  // scale 0.5, pad_top 140
    // Box ảnh gốc (100, 200)-(300, 400) -> pixel khung model -> phải về đúng chỗ cũ
    const auto d = edge::unletterbox_px(100 * lb.scale + lb.pad_left, 200 * lb.scale + lb.pad_top,
                                        300 * lb.scale + lb.pad_left, 400 * lb.scale + lb.pad_top, 0.9f, 1, lb);
    CHECK(near(d.x1, 100) && near(d.y1, 200) && near(d.x2, 300) && near(d.y2, 400));
    CHECK(d.class_id == 1 && near(d.score, 0.9f));
    // Box nằm trong vùng pad bị cắt về biên ảnh
    const auto e = edge::unletterbox_px(-20.f, 0.f, 700.f, 64.f, 0.5f, 0, lb);
    CHECK(near(e.x1, 0) && near(e.y1, 0) && near(e.y2, 0) && near(e.x2, 1280));
}

// Tensor NHWC giả lập đầu ra HEF YOLO26 cho một stride
struct FakeScale {
    int h, w, nc, stride;
    std::vector<float> box, cls;
    FakeScale(int h_, int w_, int nc_, int s_)
        : h(h_), w(w_), nc(nc_), stride(s_), box(static_cast<std::size_t>(h_ * w_ * 4), 1.f),
          cls(static_cast<std::size_t>(h_ * w_ * nc_), -20.f) {}  // logit -20 ~ score 2e-9
    void set(int y, int x, std::array<float, 4> ltrb, int c, float logit) {
        const auto cell = static_cast<std::size_t>(y * w + x);
        for (int i = 0; i < 4; ++i) box[cell * 4 + static_cast<std::size_t>(i)] = ltrb[static_cast<std::size_t>(i)];
        cls[cell * static_cast<std::size_t>(nc) + static_cast<std::size_t>(c)] = logit;
    }
    edge::ScaleOutputs view() const {
        return {stride, {box.data(), h, w, 4, box.size() * sizeof(float)},
                {cls.data(), h, w, nc, cls.size() * sizeof(float)}};
    }
};

void test_decode_end2end() {
    const auto lb = edge::compute_letterbox(640, 640, 640, 640);  // không letterbox: toạ độ model = ảnh gốc
    FakeScale p3(80, 80, 2, 8), p4(40, 40, 2, 16), p5(20, 20, 2, 32);
    p3.set(10, 20, {1.f, 2.f, 3.f, 4.f}, 1, 2.0f);   // pedestrian, score sigmoid(2) = 0.881
    p4.set(5, 6, {0.5f, 0.5f, 0.5f, 0.5f}, 0, 0.0f);  // traffic_sign, score 0.5
    p5.set(0, 0, {1.f, 1.f, 1.f, 1.f}, 0, -1.0f);     // score 0.269
    const std::vector<edge::ScaleOutputs> scales{p3.view(), p4.view(), p5.view()};

    auto dets = edge::decode_end2end(scales, 2, 0.25f, 300, lb);
    CHECK(dets.size() == 3);
    // Sắp xếp giảm dần theo score; box = (anchor -/+ ltrb) * stride, anchor = ô + 0.5
    CHECK(dets[0].class_id == 1 && near(dets[0].score, 0.880797f));
    CHECK(near(dets[0].x1, (20.5f - 1) * 8) && near(dets[0].y1, (10.5f - 2) * 8) &&
          near(dets[0].x2, (20.5f + 3) * 8) && near(dets[0].y2, (10.5f + 4) * 8));
    CHECK(dets[1].class_id == 0 && near(dets[1].score, 0.5f) && near(dets[1].x1, 6.0f * 16) && near(dets[1].y2, 6.0f * 16));
    CHECK(near(dets[2].x1, 0.f) && near(dets[2].x2, 1.5f * 32));

    CHECK(edge::decode_end2end(scales, 2, 0.6f, 300, lb).size() == 1);  // ngưỡng điểm
    CHECK(edge::decode_end2end(scales, 2, 0.25f, 2, lb).size() == 2);   // top-k (max_det)
    // conf = 0 -> mọi cặp (anchor, lớp) đều là ứng viên, top-k giữ đúng max_det
    const auto all = edge::decode_end2end(scales, 2, 0.f, 300, lb);
    CHECK(all.size() == 300 && near(all[0].score, 0.880797f));

    // Letterbox 1280x720: box được đưa về toạ độ ảnh gốc
    const auto lb169 = edge::compute_letterbox(1280, 720, 640, 640);
    const auto d169 = edge::decode_end2end(scales, 2, 0.8f, 300, lb169);
    // x: (19.5 * 8 - pad_left 0) / scale 0.5; y nằm trong vùng pad trên (68 px < pad_top 140) -> kẹp về 0
    CHECK(d169.size() == 1 && near(d169[0].x1, (20.5f - 1) * 8 * 2) && near(d169[0].x2, (20.5f + 3) * 8 * 2) &&
          near(d169[0].y1, 0.f) && near(d169[0].y2, 0.f));

    // Shape/buffer sai -> ngoại lệ, không đọc tràn
    bool threw = false;
    try {
        edge::decode_end2end(scales, 3, 0.25f, 300, lb);  // nc không khớp số kênh
    } catch (const std::runtime_error &) {
        threw = true;
    }
    CHECK(threw);
    auto truncated = p3.view();
    truncated.cls.size_bytes = 16;
    threw = false;
    try {
        edge::decode_end2end({truncated}, 2, 0.25f, 300, lb);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    CHECK(threw);
    CHECK(edge::logit_threshold(0.5f) == 0.f && std::isinf(edge::logit_threshold(0.f)));
}

void test_queue_policies() {
    edge::BoundedQueue<int> drop(2, edge::OverflowPolicy::DropOldest);
    for (int i = 0; i < 5; ++i) CHECK(drop.push(i));
    CHECK(drop.size() == 2 && drop.dropped() == 3);
    CHECK(*drop.pop() == 3 && *drop.pop() == 4);  // giữ frame mới nhất
    CHECK(!drop.pop_for(std::chrono::milliseconds(5)).has_value());
    drop.close();
    CHECK(!drop.push(9));
    CHECK(!drop.pop().has_value());

    // Block: producer chờ consumer; tất cả phần tử đến đủ và đúng thứ tự
    edge::BoundedQueue<int> block(3, edge::OverflowPolicy::Block);
    constexpr int kN = 10000;
    std::thread producer([&] {
        for (int i = 0; i < kN; ++i) block.push(i);
        block.close();
    });
    int expected = 0;
    bool in_order = true;
    while (auto v = block.pop()) {
        in_order &= (*v == expected++);
    }
    producer.join();
    CHECK(in_order && expected == kN && block.dropped() == 0);

    // close() đánh thức producer đang bị chặn vì hàng đợi đầy
    edge::BoundedQueue<int> full(1, edge::OverflowPolicy::Block);
    full.push(1);
    std::atomic<bool> returned{false};
    std::thread blocked([&] {
        const bool ok = full.push(2);
        returned = true;
        CHECK(!ok);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(!returned.load());
    full.close();
    blocked.join();
    CHECK(returned.load());
    CHECK(*full.pop() == 1 && !full.pop().has_value());  // vẫn lấy nốt phần tử còn lại
}

void test_stats() {
    edge::LatencyStats s;
    for (int i = 1; i <= 100; ++i) s.add(i);
    CHECK(near(static_cast<float>(s.mean()), 50.5f) && near(static_cast<float>(s.percentile(50)), 50.5f));
    CHECK(near(static_cast<float>(s.percentile(100)), 100.f) && near(static_cast<float>(s.percentile(0)), 1.f));
    // Chạy dài: bộ nhớ có giới hạn, trung bình vẫn chính xác trên mọi mẫu, phân vị xấp xỉ tốt
    edge::LatencyStats big(1000);
    for (int i = 0; i < 300000; ++i) big.add(i % 1000);
    CHECK(big.count() == 300000 && big.stored() == 1000);
    CHECK(near(static_cast<float>(big.mean()), 499.5f) && std::fabs(big.percentile(50) - 500.0) < 60.0);
    edge::FpsMeter m;
    auto t = edge::FpsMeter::Clock::now();
    double fps = 0;
    for (int i = 0; i < 31; ++i) fps = m.tick(t + std::chrono::milliseconds(i * 33));
    CHECK(fps > 29.0 && fps < 31.5);
}
}  // namespace

int main() {
    test_letterbox_matches_ultralytics();
    test_unletterbox_roundtrip();
    test_decode_end2end();
    test_queue_policies();
    test_stats();
    std::printf(g_failures ? "%d test(s) FAILED\n" : "All tests passed\n", g_failures);
    return g_failures ? 1 : 0;
}
