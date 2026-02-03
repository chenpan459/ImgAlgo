cmake_minimum_required(VERSION 3.10)
project(ImageComparison)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 包含OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(image_comparison image_comparison.cpp)

# 链接OpenCV库
target_link_libraries(image_comparison ${OpenCV_LIBS})

# 输出信息
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
