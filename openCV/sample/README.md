# OpenCV 入门学习 Demo

本目录是面向 `opencv-4.13.0` 的**实践入门路径**。API 与系统已安装的 OpenCV 4.x（当前环境多为 4.5.4）兼容；源码树在 `../opencv-4.13.0/`，理论文档在 `../doc/`。

## 怎么入门（建议顺序）

```text
第 0 周：环境 + 跑通第一个程序
第 1 周：Mat / 颜色 / 绘图 / 滤波（01–05）入门
第 2 周：边缘 / 形态学 / 轮廓 / 几何 / ORB（06–10）入门
第 3 周：直方图 / 霍夫 / 模板 / 单应性 / 连通域 / 光流 / 背景（11–17）中级
第 4 周：分水岭 / GrabCut / 拼接 / ArUco / 卡尔曼 / 文档扫描 / 立体 / 标定 / 修复（18–26）高级
第 5 周：DiffImg + 真实摄像头 / DNN / 结合 ultralytics
```

### 配合阅读

| 顺序 | 材料 | 作用 |
|------|------|------|
| 1 | 本 README + 各 demo 源码头注释 | 动手主线 |
| 2 | [`各示例所需基础知识.md`](./各示例所需基础知识.md) | **每个例子的前置知识 / 前置 Demo** |
| 3 | [`../doc/OpenCV_4.13.0_快速参考.md`](../doc/OpenCV_4.13.0_快速参考.md) | API 速查 |
| 4 | [`../doc/OpenCV_4.13.0_代码结构分析.md`](../doc/OpenCV_4.13.0_代码结构分析.md) | 源码模块地图 |
| 5 | [`../doc/OpenCV_4.13.0_图片算法详解.md`](../doc/OpenCV_4.13.0_图片算法详解.md) | 算法原理加深 |
| 6 | 官方教程 https://docs.opencv.org/4.x/d9/df8/tutorial_root.html | 系统补强 |

### 学习原则

1. **先跑再改**：每个 demo 先默认运行，再改参数（如 `--canny_low`）看变化。  
2. **先看输出图**：无显示器时看 `output/`；有 GUI 加 `--show`。  
3. **对照模块**：`core`→Mat，`imgproc`→滤波/边缘/轮廓，`features2d`→ORB。  
4. **少而精**：入门先掌握下表 10 个 demo，再进官方 `opencv-4.13.0/samples/cpp`。

## Demo 清单

| Demo | 目录 | 知识点 | 对应模块 |
|------|------|--------|----------|
| 01 | `01_hello_image` | imread/imwrite、灰度 | core, imgcodecs, imgproc |
| 02 | `02_mat_roi` | 像素访问、ROI 引用 vs clone | core |
| 03 | `03_color_space` | BGR/HSV、颜色阈值 | imgproc |
| 04 | `04_drawing` | 画线框圆字 | imgproc |
| 05 | `05_filter` | 均值/高斯/中值/双边 | imgproc |
| 06 | `06_edge` | Sobel / Laplacian / Canny | imgproc |
| 07 | `07_morphology` | 腐蚀膨胀开闭 | imgproc |
| 08 | `08_contour` | 轮廓、外接矩形、矩 | imgproc |
| 09 | `09_geometry` | 缩放翻转仿射透视 | imgproc |
| 10 | `10_feature_match` | ORB + BFMatcher | features2d |

### 中级（11–17）

| Demo | 目录 | 知识点 | 对应模块 |
|------|------|--------|----------|
| 11 | `11_histogram` | 直方图、equalizeHist、CLAHE | imgproc |
| 12 | `12_hough` | HoughLinesP / HoughCircles | imgproc |
| 13 | `13_template_match` | matchTemplate、多尺度 | imgproc |
| 14 | `14_homography` | ORB + findHomography 对齐 | features2d, calib3d |
| 15 | `15_connected_components` | 连通域统计与过滤 | imgproc |
| 16 | `16_optical_flow` | LK 稀疏 + Farneback 稠密光流 | video |
| 17 | `17_bg_subtract` | MOG2 背景建模（合成视频） | video |

### 高级（18–26）

| Demo | 目录 | 知识点 | 对应模块 |
|------|------|--------|----------|
| 18 | `18_watershed` | 分水岭分割 | imgproc |
| 19 | `19_grabcut` | GrabCut 前景提取 | imgproc |
| 20 | `20_stitching` | Stitcher 全景拼接 | stitching |
| 21 | `21_aruco` | ArUco 检测与位姿 | aruco, calib3d |
| 22 | `22_kalman_track` | 卡尔曼滤波跟踪 | video |
| 23 | `23_document_scan` | 文档四边形拉正流水线 | imgproc, calib3d |
| 24 | `24_stereo_bm` | StereoBM 视差/深度 | calib3d |
| 25 | `25_camera_calib` | 棋盘格相机标定 | calib3d |
| 26 | `26_inpaint_clone` | inpaint + seamlessClone | photo |

### 综合进阶

| Demo | 目录 | 知识点 | 对应模块 |
|------|------|--------|----------|
| 工具 | `DiffImg` | SSIM / pHash / 相似度 | imgproc, features2d |

## 编译

依赖：CMake ≥ 3.10，g++ 支持 C++17，已安装 `libopencv-dev`（或自编译 OpenCV）。

推荐使用脚本：

```bash
cd /home/cp/work2/visualAlgo/openCV/sample
chmod +x build.sh
./build.sh              # 配置并编译
./build.sh clean        # 清理后重编
./build.sh run          # 入门 01–10
./build.sh run-mid      # 中级 11–17
./build.sh run-adv      # 高级 18–26
./build.sh run-all      # 全部 01–26
./build.sh -j8
./build.sh --opencv-dir /path/to/lib/cmake/opencv4
./build.sh --debug
```

手动 CMake：

```bash
mkdir -p build && cd build
cmake ..
cmake --build . -j$(nproc)
```

若使用自行编译的 OpenCV 4.13：

```bash
./build.sh --opencv-dir /path/to/opencv-4.13.0/install/lib/cmake/opencv4
# 或
cmake -DOpenCV_DIR=/path/to/opencv-4.13.0/install/lib/cmake/opencv4 ..
```

单独编译 DiffImg：

```bash
cd DiffImg && chmod +x build.sh && ./build.sh
```

## 运行

可执行文件在 `build/bin/`。默认写结果到 `sample/output/<demo名>/`，**不弹窗**。

```bash
cd /home/cp/work2/visualAlgo/openCV/sample/build

# 单个 demo（合成测试图）
./bin/01_hello_image --outdir ../output

# 使用自己的图片
./bin/05_filter --image /path/to/photo.jpg --outdir ../output

# 有显示器时弹窗
./bin/06_edge --show --canny_low=30 --canny_high=100

# 一键跑
cmake --build . --target run_beginner_demos   # 01-10
cmake --build . --target run_mid_demos        # 11-17
cmake --build . --target run_adv_demos        # 18-26
cmake --build . --target run_all_demos        # 全部
```

通用参数：

| 参数 | 含义 |
|------|------|
| `--image path` 或首个位置参数 | 输入图；省略则用合成图 |
| `--outdir dir` | 输出根目录，默认 `output` |
| `--show` | `imshow` 显示（需 GUI） |

## 每日练习建议（约 7–10 天）

| 天 | 任务 |
|----|------|
| Day 1 | 编译通过；跑 `01`，弄清 BGR 与 channels |
| Day 2 | `02`：故意改 ROI，观察引用副作用 |
| Day 3 | `03`：换一张图，调 `inRange` 阈值别的颜色 |
| Day 4 | `04`+`05`：给自己照片加标注再滤波对比 |
| Day 5 | `06`+`07`：调 Canny 阈值与核大小 |
| Day 6 | `08`：过滤小轮廓，打印面积最大轮廓 |
| Day 7 | `09`：改透视四点，做“文档拉正”实验 |
| Day 8 | `10`：换两张相似图，看匹配数变化 |
| Day 9–10 | 中级 `11`–`13`：直方图 / 霍夫 / 模板匹配 |
| Day 11–12 | 中级 `14`–`15`：单应性对齐 / 连通域 |
| Day 13–14 | 中级 `16`–`17`：光流 / 背景减除 |
| Day 15–17 | 高级 `18`–`21`：分水岭 / GrabCut / 拼接 / ArUco |
| Day 18–20 | 高级 `22`–`26`：卡尔曼 / 文档扫描 / 立体 / 标定 / 修复 |
| Day 21+ | `DiffImg` + 摄像头 + DNN + `doc` 算法文档 |

## 摄像头小练习（可选，自写）

```cpp
cv::VideoCapture cap(0);
cv::Mat frame;
while (cap.read(frame)) {
    cv::imshow("cam", frame);
    if (cv::waitKey(1) == 27) break; // ESC
}
```

## 源码树怎么用

不必一上来通读 `opencv-4.13.0/`。按 demo 对照：

```text
modules/core      ← Mat、并行、基础类型
modules/imgproc   ← 滤波、边缘、轮廓、几何、颜色
modules/imgcodecs ← imread/imwrite
modules/highgui   ← imshow/waitKey
modules/features2d← ORB、匹配器
modules/dnn       ← 深度学习推理（进阶）
modules/calib3d   ← 标定、立体视觉（进阶）
modules/videoio   ← 摄像头/视频（进阶）
```

官方更多示例：`opencv-4.13.0/samples/cpp/`。

## 目录结构

```text
sample/
├── README.md                 # 本学习指南
├── CMakeLists.txt            # 统一构建
├── common/demo_utils.hpp     # 公共工具
├── 01_hello_image/ … 10_feature_match/   # 入门
├── 11_histogram/ … 17_bg_subtract/       # 中级
├── 18_watershed/ … 26_inpaint_clone/     # 高级
├── DiffImg/                              # 相似度工具
└── output/                               # 运行产物
```
