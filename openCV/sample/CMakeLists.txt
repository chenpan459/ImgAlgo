cmake_minimum_required(VERSION 3.10)
project(opencv_learning_samples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 优先使用本机已安装的 OpenCV（Ubuntu 常见为 4.5.x，API 与 4.13 入门用法兼容）
# 若已从源码安装 4.13，可：cmake -DOpenCV_DIR=/path/to/lib/cmake/opencv4 ..
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV ${OpenCV_VERSION}")
message(STATUS "OpenCV_DIR=${OpenCV_DIR}")
message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/common)

set(DEMO_LIST
    01_hello_image
    02_mat_roi
    03_color_space
    04_drawing
    05_filter
    06_edge
    07_morphology
    08_contour
    09_geometry
    10_feature_match
)

foreach(demo ${DEMO_LIST})
    add_executable(${demo} ${demo}/main.cpp)
    target_link_libraries(${demo} PRIVATE ${OpenCV_LIBS})
    # 运行时默认把 output 写到 build 目录上级 sample/
    set_target_properties(${demo} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()

# 进阶：图像相似度（原有 DiffImg）
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/DiffImg/image_comparison.cpp)
    add_executable(image_comparison DiffImg/image_comparison.cpp)
    target_link_libraries(image_comparison PRIVATE ${OpenCV_LIBS})
    set_target_properties(image_comparison PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# 一键跑完全部入门 demo（无 GUI）
add_custom_target(run_all_demos
    COMMAND ${CMAKE_COMMAND} -E echo "Running all beginner demos..."
    COMMAND $<TARGET_FILE:01_hello_image> --outdir ${CMAKE_CURRENT_SOURCE_DIR}/output
    COMMAND $<TARGET_FILE:02_mat_roi> --outdir ${CMAKE_CURRENT_SOURCE_DIR}/output
    COMMAND $<TARGET_FILE:03_color_space> --outdir ${CMAKE_CURRENT_SOURCE_DIR}/output
    COMMAND $<TARGET_FILE:04_drawing> --outdir ${CMAKE_CURRENT_SOURCE_DIR}/output
    COMMAND $<TARGET_FILE:05_filter> --outdir ${CMAKE_CURRENT_SOURCE_DIR}/output
    COMMAND $<TARGET_FILE:06_edge> --outdir ${CMAKE_CURRENT_SOURCE_DIR}/output
    COMMAND $<TARGET_FILE:07_morphology> --outdir ${CMAKE_CURRENT_SOURCE_DIR}/output
    COMMAND $<TARGET_FILE:08_contour> --outdir ${CMAKE_CURRENT_SOURCE_DIR}/output
    COMMAND $<TARGET_FILE:09_geometry> --outdir ${CMAKE_CURRENT_SOURCE_DIR}/output
    COMMAND $<TARGET_FILE:10_feature_match> --outdir ${CMAKE_CURRENT_SOURCE_DIR}/output
    DEPENDS ${DEMO_LIST}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "Run 01-10 demos and write images to sample/output/"
)
