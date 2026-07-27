# OpenCV 4.13.0 入门学习路径

配套实践代码：[`../sample/`](../sample/README.md)

## 目标

用最短路径掌握 OpenCV 日常 80% 用法：读图、Mat、颜色、滤波、边缘、轮廓、几何变换、特征匹配；并能对照源码模块继续深入。

## 环境说明

| 项 | 说明 |
|----|------|
| 源码 | `openCV/opencv-4.13.0/` |
| 本机开发库 | 可用系统 `libopencv-dev`（如 4.5.4），入门 API 兼容 |
| 实践目录 | `openCV/sample/`（01–10 + DiffImg） |
| 理论文档 | 本目录下结构分析 / 算法详解 / 快速参考 |

## 三阶段路线

### 阶段 A：会用（1–2 周）入门 01–10

按顺序完成 `sample` 中 01→10，每天 1–2 个 demo：

1. Hello Image → Mat ROI → Color → Drawing → Filter  
2. Edge → Morphology → Contour → Geometry → Feature Match  

完成标准：每个 demo 能解释**输入、关键 API、输出图差异**。

### 阶段 B：中级应用（1–2 周）11–17

```bash
./build.sh run-mid
```

| Demo | 能力 |
|------|------|
| 11_histogram | 对比度增强、CLAHE |
| 12_hough | 直线/圆检测 |
| 13_template_match | 模板与多尺度搜索 |
| 14_homography | 特征对齐 / 透视校正基础 |
| 15_connected_components | 连通域测量与筛选 |
| 16_optical_flow | 运动估计（LK / Farneback） |
| 17_bg_subtract | 运动前景（MOG2） |

完成标准：能把中级算法**接到自己的图片或短视频流程**里。

### 阶段 C：高级能力（1–2 周）18–26

```bash
./build.sh run-adv
```

| Demo | 能力 |
|------|------|
| 18_watershed | 标记分水岭分割 |
| 19_grabcut | 交互式前景抠图基础 |
| 20_stitching | 全景拼接 |
| 21_aruco | 标记检测与位姿估计 |
| 22_kalman_track | 状态估计跟踪 |
| 23_document_scan | 端到端文档矫正 |
| 24_stereo_bm | 双目视差 |
| 25_camera_calib | 相机内参标定 |
| 26_inpaint_clone | 图像修复与无缝融合 |
| DiffImg | 图像相似度（SSIM/pHash） |

### 阶段 D：能改（1 周）

- 用自己的图片替换合成图（`--image`）  
- 改阈值/核大小，记录现象  
- 把 `08_contour` / `15` 改成“只保留最大区域”  
- 阅读 `OpenCV_4.13.0_快速参考.md`

### 阶段 E：能挖（持续）

- 读 `OpenCV_4.13.0_代码结构分析.md`，定位 `imgproc` / `features2d` / `video` / `calib3d` / `dnn`  
- 跑官方 `opencv-4.13.0/samples/cpp` 中 calibration、stitching、dnn  
- 真实摄像头 `VideoCapture` + ArUco/标定  
- 与 `ultralytics` 结合做检测后处理  

## 模块 ↔ Demo 映射

| 模块 | Demo | 先掌握的符号 |
|------|------|----------------|
| core | 01, 02 | `Mat`, `at`, `ptr`, `Rect`, `clone` |
| imgcodecs / highgui | 01 | `imread`, `imwrite`, `imshow` |
| imgproc | 03–09 | `cvtColor`, `GaussianBlur`, `Canny`, `findContours`, `warpAffine` |
| features2d | 10, DiffImg | `ORB`, `BFMatcher`, `drawMatches` |

## 推荐时间表（10 天）

见 [`../sample/README.md`](../sample/README.md)「每日练习建议」。

## 常见坑

1. **颜色通道是 BGR**，不是 RGB。  
2. **ROI 默认共享数据**，要独立拷贝用 `clone()`。  
3. **无 GUI 环境**不要依赖 `imshow`，用 `--outdir` 写文件（demo 已默认如此）。  
4. **Canny 阈值**随图像尺度变化大，先高斯再调。  
5. 系统 OpenCV 与源码 4.13 **版本号可不同**，入门练习无影响；用新 API 时再切到自编译 4.13。

## 下一步项目想法

- 文档扫描：边缘 + 轮廓 + 透视拉正（复用 06/08/09）  
- 简易相似度工具：扩展 DiffImg  
- 摄像头实时 Canny / 颜色跟踪  
- DNN：用 `opencv_zoo` 或 ONNX 跑检测，结果画框（`04_drawing`）
