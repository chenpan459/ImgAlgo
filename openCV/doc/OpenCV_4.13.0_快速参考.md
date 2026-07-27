# OpenCV 4.13.0 快速参考指南

## 目录结构快速查找

### 核心代码位置
```
opencv-4.13.0/
├── modules/              # 所有功能模块
│   ├── core/            # 核心模块
│   ├── imgproc/         # 图像处理
│   ├── imgcodecs/       # 图像编解码
│   ├── dnn/             # 深度学习
│   └── ...
├── include/opencv2/     # 公共头文件
├── samples/             # 示例代码
└── doc/                 # 文档
```

### 常用头文件
```cpp
#include <opencv2/opencv.hpp>        // 包含所有模块
#include <opencv2/core.hpp>          // 核心功能
#include <opencv2/imgproc.hpp>       // 图像处理
#include <opencv2/imgcodecs.hpp>     // 图像编解码
#include <opencv2/highgui.hpp>       // GUI 显示
#include <opencv2/dnn.hpp>           // 深度学习
#include <opencv2/objdetect.hpp>     // 目标检测
#include <opencv2/features2d.hpp>    // 特征检测
```

## 主要模块功能速查

### Core 模块
| 功能 | 主要类/函数 |
|------|-----------|
| 矩阵操作 | `cv::Mat`, `cv::Mat_` |
| 数据类型 | `cv::Point`, `cv::Size`, `cv::Rect`, `cv::Scalar` |
| 文件操作 | `cv::FileStorage`, `cv::FileNode` |
| 并行处理 | `cv::parallel_for_` |

### Imgproc 模块
| 功能 | 主要函数 |
|------|---------|
| 滤波 | `cv::GaussianBlur()`, `cv::medianBlur()`, `cv::bilateralFilter()` |
| 几何变换 | `cv::resize()`, `cv::warpAffine()`, `cv::warpPerspective()` |
| 形态学 | `cv::erode()`, `cv::dilate()`, `cv::morphologyEx()` |
| 边缘检测 | `cv::Canny()`, `cv::Sobel()`, `cv::Laplacian()` |
| 轮廓 | `cv::findContours()`, `cv::drawContours()` |
| 直方图 | `cv::calcHist()`, `cv::equalizeHist()` |

### Imgcodecs 模块
| 功能 | 主要函数 |
|------|---------|
| 读取图像 | `cv::imread()` |
| 保存图像 | `cv::imwrite()` |
| 图像格式 | JPEG, PNG, TIFF, WebP, OpenEXR 等 |

### Highgui 模块
| 功能 | 主要函数 |
|------|---------|
| 显示图像 | `cv::imshow()`, `cv::namedWindow()` |
| 等待按键 | `cv::waitKey()` |
| 鼠标回调 | `cv::setMouseCallback()` |
| 滑动条 | `cv::createTrackbar()` |

### DNN 模块
| 功能 | 主要类/函数 |
|------|-----------|
| 加载模型 | `cv::dnn::readNet()`, `cv::dnn::readNetFromONNX()` |
| 推理 | `net.forward()`, `net.setInput()` |
| 后端 | CPU, CUDA, OpenCL |

### Objdetect 模块
| 功能 | 主要类/函数 |
|------|-----------|
| 人脸检测 | `cv::CascadeClassifier` |
| QR 码 | `cv::QRCodeDetector` |
| HOG 检测 | `cv::HOGDescriptor` |

### Features2d 模块
| 功能 | 主要类 |
|------|-------|
| 特征检测 | `cv::SIFT`, `cv::SURF`, `cv::ORB`, `cv::AKAZE` |
| 特征匹配 | `cv::BFMatcher`, `cv::FlannBasedMatcher` |

### Calib3d 模块
| 功能 | 主要函数 |
|------|---------|
| 相机标定 | `cv::calibrateCamera()` |
| 立体标定 | `cv::stereoCalibrate()` |
| PnP 求解 | `cv::solvePnP()` |

### Video 模块
| 功能 | 主要类/函数 |
|------|-----------|
| 光流 | `cv::calcOpticalFlowPyrLK()`, `cv::calcOpticalFlowFarneback()` |
| 背景减除 | `cv::createBackgroundSubtractorMOG2()` |
| 跟踪 | `cv::TrackerKCF`, `cv::TrackerCSRT` |

## 常用代码模式

### 读取和显示图像
```cpp
#include <opencv2/opencv.hpp>
using namespace cv;

Mat img = imread("image.jpg");
imshow("Window", img);
waitKey(0);
```

### 图像处理流程
```cpp
Mat img = imread("image.jpg");
Mat gray, blurred, edges;
cvtColor(img, gray, COLOR_BGR2GRAY);
GaussianBlur(gray, blurred, Size(5,5), 0);
Canny(blurred, edges, 50, 150);
imshow("Edges", edges);
waitKey(0);
```

### 特征检测和匹配
```cpp
Ptr<ORB> detector = ORB::create();
vector<KeyPoint> keypoints;
Mat descriptors;
detector->detectAndCompute(img, noArray(), keypoints, descriptors);

BFMatcher matcher(NORM_HAMMING);
vector<DMatch> matches;
matcher.match(descriptors1, descriptors2, matches);
```

### DNN 推理
```cpp
Net net = dnn::readNetFromONNX("model.onnx");
Mat blob = dnn::blobFromImage(img, 1.0/255.0, Size(224,224));
net.setInput(blob);
Mat output = net.forward();
```

### 相机标定
```cpp
vector<vector<Point3f>> objectPoints;
vector<vector<Point2f>> imagePoints;
Mat cameraMatrix, distCoeffs;
calibrateCamera(objectPoints, imagePoints, imageSize, 
                cameraMatrix, distCoeffs, rvecs, tvecs);
```

## 文件查找指南

### 查找特定功能实现
1. **图像处理算法**: `modules/imgproc/src/`
2. **深度学习层**: `modules/dnn/src/layers/`
3. **特征检测器**: `modules/features2d/src/`
4. **目标检测**: `modules/objdetect/src/`
5. **相机标定**: `modules/calib3d/src/`

### 查找示例代码
1. **C++ 示例**: `samples/cpp/`
2. **Python 示例**: `samples/python/`
3. **特定模块示例**: `samples/cpp/tutorial_code/`

### 查找测试代码
1. **单元测试**: `modules/[module_name]/test/`
2. **性能测试**: `modules/[module_name]/perf/`

### 查找文档
1. **C++ 教程**: `doc/tutorials/`
2. **Python 教程**: `doc/py_tutorials/`
3. **JavaScript 教程**: `doc/js_tutorials/`

## 编译配置速查

### 最小配置（CMake）
```cmake
cmake_minimum_required(VERSION 3.5)
project(OpenCVTest)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(your_target ${OpenCV_LIBS})
```

### 常用 CMake 选项
```bash
# 禁用不需要的模块
cmake -D BUILD_opencv_java=OFF
cmake -D BUILD_opencv_js=OFF

# 启用 GPU 支持
cmake -D WITH_CUDA=ON
cmake -D WITH_OPENCL=ON

# 指定安装路径
cmake -D CMAKE_INSTALL_PREFIX=/usr/local
```

## 常见问题快速解决

### 问题：找不到头文件
**解决**: 检查 `include_directories` 是否包含 OpenCV 头文件路径

### 问题：链接错误
**解决**: 确保链接了所有需要的 OpenCV 库

### 问题：图像读取失败
**解决**: 
- 检查文件路径是否正确
- 检查图像格式是否支持
- 检查 `imgcodecs` 模块是否编译

### 问题：DNN 模型加载失败
**解决**:
- 检查模型格式是否支持
- 检查 protobuf 是否正确安装
- 检查模型文件路径

## 性能优化提示

### 1. 使用 SIMD 优化
- 编译时启用 SSE、AVX 等指令集
- 使用 `cv::parallel_for_` 进行并行处理

### 2. 内存管理
- 避免不必要的图像复制
- 使用 `Mat::clone()` 仅在需要时

### 3. GPU 加速
- 使用 CUDA 后端（NVIDIA GPU）
- 使用 OpenCL 后端（跨平台）

### 4. 数据类型选择
- 使用合适的图像数据类型（CV_8U, CV_32F 等）
- 避免不必要的类型转换

## 版本兼容性

### OpenCV 4.x 主要变化
- 移除了 C API（cv* 函数）
- 统一使用 C++ API
- 模块化程度更高
- DNN 模块功能增强

### 从 OpenCV 3.x 迁移
- 更新头文件包含方式
- 更新命名空间使用
- 检查已废弃的函数

## 资源链接

### 官方资源
- 文档: https://docs.opencv.org/4.x/
- 教程: https://docs.opencv.org/4.x/d9/df8/tutorial_root.html
- API 参考: https://docs.opencv.org/4.x/db/d4d/group__core__basic.html

### 社区资源
- 论坛: https://forum.opencv.org
- GitHub: https://github.com/opencv/opencv
- Stack Overflow: 搜索 "opencv" 标签

---

*快速参考指南 - OpenCV 4.13.0*
