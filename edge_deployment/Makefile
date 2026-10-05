# Makefile - lối tắt cho CMake + đo kiểm (chạy trong thư mục edge_deployment/ trên Raspberry Pi 5)
#   make            build inference_app (Release)
#   make test       build + chạy unit test (không cần Hailo-8)
#   make run        chạy camera USB /dev/video0     | make run SOURCE=libcamera | make run SOURCE=video.mp4
#   make bench      đo kiểm đầy đủ (mAP + NPU + stress 5 phút) bằng benchmark_profiler.py

BUILD_DIR ?= build
JOBS      ?= 4
HEF       ?= ../models/best_hailo.hef
ONNX      ?= ../models/best.onnx
SOURCE    ?= 0
ARGS      ?=
PYTHON    ?= python3

.PHONY: all build test run headless bench accuracy stress npu clean

all: build

build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) -j$(JOBS)

test:
	cmake -S . -B $(BUILD_DIR)-test -DBKAUTO_BUILD_APP=OFF -DBKAUTO_BUILD_TESTS=ON
	cmake --build $(BUILD_DIR)-test -j$(JOBS)
	ctest --test-dir $(BUILD_DIR)-test --output-on-failure

run: build
	./$(BUILD_DIR)/inference_app --hef $(HEF) --source $(SOURCE) $(ARGS)

headless: build
	./$(BUILD_DIR)/inference_app --hef $(HEF) --source $(SOURCE) --headless --stats frame_stats.csv $(ARGS)

accuracy:
	$(PYTHON) benchmark_profiler.py accuracy --hef $(HEF)

npu:
	$(PYTHON) benchmark_profiler.py npu --hef $(HEF)

stress: build
	$(PYTHON) benchmark_profiler.py stress --engine cpp --hef $(HEF) --source $(SOURCE)

bench: build
	$(PYTHON) benchmark_profiler.py all --hef $(HEF) --onnx $(ONNX) --engine cpp --source $(SOURCE)

clean:
	rm -rf $(BUILD_DIR) $(BUILD_DIR)-test
