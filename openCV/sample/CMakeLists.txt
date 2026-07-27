cmake_minimum_required(VERSION 3.10)
project(opencv_learning_samples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV ${OpenCV_VERSION}")
message(STATUS "OpenCV_DIR=${OpenCV_DIR}")
message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/common)

set(DEMO_BEGINNER
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

set(DEMO_INTERMEDIATE
    11_histogram
    12_hough
    13_template_match
    14_homography
    15_connected_components
    16_optical_flow
    17_bg_subtract
)

set(DEMO_ADVANCED
    18_watershed
    19_grabcut
    20_stitching
    21_aruco
    22_kalman_track
    23_document_scan
    24_stereo_bm
    25_camera_calib
    26_inpaint_clone
)

set(DEMO_LIST ${DEMO_BEGINNER} ${DEMO_INTERMEDIATE} ${DEMO_ADVANCED})

foreach(demo ${DEMO_LIST})
    add_executable(${demo} ${demo}/main.cpp)
    target_link_libraries(${demo} PRIVATE ${OpenCV_LIBS})
    set_target_properties(${demo} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/DiffImg/image_comparison.cpp)
    add_executable(image_comparison DiffImg/image_comparison.cpp)
    target_link_libraries(image_comparison PRIVATE ${OpenCV_LIBS})
    set_target_properties(image_comparison PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

set(OUTDIR ${CMAKE_CURRENT_SOURCE_DIR}/output)

add_custom_target(run_beginner_demos
    COMMAND ${CMAKE_COMMAND} -E echo "Running beginner demos 01-10..."
    COMMAND $<TARGET_FILE:01_hello_image> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:02_mat_roi> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:03_color_space> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:04_drawing> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:05_filter> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:06_edge> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:07_morphology> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:08_contour> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:09_geometry> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:10_feature_match> --outdir ${OUTDIR}
    DEPENDS ${DEMO_BEGINNER}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "Run beginner demos 01-10"
)

add_custom_target(run_mid_demos
    COMMAND ${CMAKE_COMMAND} -E echo "Running intermediate demos 11-17..."
    COMMAND $<TARGET_FILE:11_histogram> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:12_hough> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:13_template_match> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:14_homography> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:15_connected_components> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:16_optical_flow> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:17_bg_subtract> --outdir ${OUTDIR}
    DEPENDS ${DEMO_INTERMEDIATE}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "Run intermediate demos 11-17"
)

add_custom_target(run_adv_demos
    COMMAND ${CMAKE_COMMAND} -E echo "Running advanced demos 18-26..."
    COMMAND $<TARGET_FILE:18_watershed> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:19_grabcut> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:20_stitching> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:21_aruco> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:22_kalman_track> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:23_document_scan> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:24_stereo_bm> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:25_camera_calib> --outdir ${OUTDIR}
    COMMAND $<TARGET_FILE:26_inpaint_clone> --outdir ${OUTDIR}
    DEPENDS ${DEMO_ADVANCED}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "Run advanced demos 18-26"
)

add_custom_target(run_all_demos
    COMMAND ${CMAKE_COMMAND} -E echo "Running all demos 01-26..."
    DEPENDS run_beginner_demos run_mid_demos run_adv_demos
    COMMENT "Run beginner + intermediate + advanced demos"
)
