# CMakeLists.txt - inference_app (YOLO26m NMS-free trên Raspberry Pi 5 + Hailo-8)
#
# Build trên Pi (Raspberry Pi OS 64-bit, sau khi `sudo apt install hailo-all libopencv-dev cmake g++`):
#   cmake -S edge_deployment -B edge_deployment/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build edge_deployment/build -j4
# Unit test không cần HailoRT/OpenCV:
#   cmake -S edge_deployment -B edge_deployment/build-test -DBKAUTO_BUILD_APP=OFF -DBKAUTO_BUILD_TESTS=ON
#   cmake --build edge_deployment/build-test && ctest --test-dir edge_deployment/build-test
cmake_minimum_required(VERSION 3.16)
project(bkauto_edge VERSION 1.0 LANGUAGES CXX)

# ---- Tuỳ chọn ---------------------------------------------------------------------------------
option(BKAUTO_BUILD_APP "Build inference_app (cần HailoRT + OpenCV)" ON)
option(BKAUTO_BUILD_TESTS "Build unit test cho include/detection_utils.hpp" OFF)
option(BKAUTO_NATIVE "Tối ưu cho CPU của máy build (-mcpu=native; Cortex-A76 khi build trên Pi 5)" ON)
set(BKAUTO_MIN_HAILORT_VERSION 4.20 CACHE STRING "HailoRT tối thiểu (InferModel API, đầu ra FLOAT32 NHWC)")

# ---- Chuẩn C++ & kiểu build ----------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Debug | Release | RelWithDebInfo | MinSizeRel" FORCE)
endif()

find_package(Threads REQUIRED)

# Cờ cảnh báo/tối ưu dùng chung, gắn theo target (không đổi CMAKE_CXX_FLAGS toàn cục)
add_library(bkauto_options INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bkauto_options INTERFACE -Wall -Wextra -Wpedantic)
    # -mcpu=native chỉ đúng khi build ngay trên Pi; khi cross-compile "native" là CPU máy build
    if(BKAUTO_NATIVE AND NOT CMAKE_CROSSCOMPILING)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-mcpu=native BKAUTO_HAS_MCPU_NATIVE)
        if(BKAUTO_HAS_MCPU_NATIVE)
            target_compile_options(bkauto_options INTERFACE
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-mcpu=native>)
        endif()
    endif()
endif()

# Thành phần C++17 thuần (hàng đợi, giải mã NMS, letterbox) - header-only
add_library(detection_utils INTERFACE)
target_include_directories(detection_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(detection_utils INTERFACE Threads::Threads)

# ---- Ứng dụng --------------------------------------------------------------------------------
if(BKAUTO_BUILD_APP)
    # HailoRT: gói `hailort` (cài qua hailo-all) cung cấp /usr/lib/cmake/HailoRT/HailoRTConfig.cmake với
    # target HailoRT::libhailort. KHÔNG ghi version trong find_package: HailoRTConfigVersion.cmake dùng
    # chế độ ExactVersion nên yêu cầu "4.20" bị từ chối kể cả khi cài 4.20.0 -> tự so sánh phiên bản.
    find_package(HailoRT CONFIG QUIET)
    if(NOT HailoRT_FOUND)
        message(FATAL_ERROR
            "Không tìm thấy HailoRT. Trên Raspberry Pi: `sudo apt install hailo-all`.\n"
            "Nếu cài ở thư mục khác: -DHailoRT_DIR=<prefix>/lib/cmake/HailoRT")
    endif()
    if(HailoRT_VERSION VERSION_LESS BKAUTO_MIN_HAILORT_VERSION)
        message(FATAL_ERROR "Cần HailoRT >= ${BKAUTO_MIN_HAILORT_VERSION}, tìm thấy ${HailoRT_VERSION}")
    endif()

    # OpenCV 4.x: imported target cho từng module, đã mang sẵn include dir và thư viện phụ thuộc
    find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)

    message(STATUS "HailoRT ${HailoRT_VERSION} (${HailoRT_DIR})")
    message(STATUS "OpenCV  ${OpenCV_VERSION} (${OpenCV_DIR})")

    add_executable(inference_app inference_app.cpp)
    target_link_libraries(inference_app
        PRIVATE
            detection_utils
            bkauto_options
            HailoRT::libhailort
            opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_highgui
            Threads::Threads)
    install(TARGETS inference_app RUNTIME DESTINATION bin)
endif()

# ---- Unit test -------------------------------------------------------------------------------
if(BKAUTO_BUILD_TESTS)
    enable_testing()
    add_executable(test_detection_utils tests/test_detection_utils.cpp)
    target_link_libraries(test_detection_utils PRIVATE detection_utils bkauto_options)
    add_test(NAME detection_utils COMMAND test_detection_utils)
endif()
