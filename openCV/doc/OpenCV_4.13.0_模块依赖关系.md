# OpenCV 4.13.0 模块依赖关系图

## 模块依赖层次结构

```
┌─────────────────────────────────────────────────────────────┐
│                        Core Module                           │
│  (基础数据结构、内存管理、数学运算、并行处理)                  │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Imgcodecs   │    │   Imgproc    │    │   Highgui    │
│ (图像编解码)  │    │  (图像处理)   │    │  (GUI显示)   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Features2d   │    │   Calib3d     │    │  Objdetect   │
│ (特征检测)   │    │  (相机标定)    │    │  (目标检测)   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Stitching  │    │    Video     │    │    Photo     │
│  (图像拼接)   │    │  (视频分析)   │    │ (计算摄影)   │
└──────────────┘    └──────────────┘    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │     DNN      │
                    │  (深度学习)   │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │     GAPI     │
                    │  (图计算框架) │
                    └──────────────┘
```

## 详细依赖关系

### Core 模块
**依赖**: 无（基础模块）
**被依赖**: 所有其他模块

### Imgcodecs 模块
**依赖**: 
- Core

**被依赖**:
- Imgproc
- Highgui
- 所有需要读取图像的模块

### Imgproc 模块
**依赖**: 
- Core
- Imgcodecs

**被依赖**:
- Features2d
- Calib3d
- Objdetect
- Photo
- Stitching
- Video
- DNN

### Highgui 模块
**依赖**: 
- Core
- Imgcodecs

**被依赖**:
- 所有需要显示图像的模块

### Features2d 模块
**依赖**: 
- Core
- Imgproc
- Flann（可选）

**被依赖**:
- Calib3d
- Stitching

### Calib3d 模块
**依赖**: 
- Core
- Imgproc
- Features2d

**被依赖**:
- 无（终端模块）

### Objdetect 模块
**依赖**: 
- Core
- Imgproc

**被依赖**:
- 无（终端模块）

### Video 模块
**依赖**: 
- Core
- Imgproc

**被依赖**:
- 无（终端模块）

### Videoio 模块
**依赖**: 
- Core

**被依赖**:
- Video
- Highgui

### Photo 模块
**依赖**: 
- Core
- Imgproc

**被依赖**:
- 无（终端模块）

### Stitching 模块
**依赖**: 
- Core
- Imgproc
- Features2d
- Calib3d

**被依赖**:
- 无（终端模块）

### DNN 模块
**依赖**: 
- Core
- Imgproc

**被依赖**:
- Objdetect（可选，用于 DNN 目标检测）
- GAPI（可选）

### ML 模块
**依赖**: 
- Core

**被依赖**:
- 无（独立模块）

### GAPI 模块
**依赖**: 
- Core
- Imgproc
- DNN（可选）

**被依赖**:
- 无（终端模块）

### Flann 模块
**依赖**: 
- Core

**被依赖**:
- Features2d

## 模块分组

### 基础模块组
- **Core**: 核心功能
- **Imgcodecs**: 图像编解码
- **Imgproc**: 图像处理
- **Highgui**: GUI 显示

### 计算机视觉模块组
- **Features2d**: 特征检测
- **Calib3d**: 相机标定
- **Objdetect**: 目标检测
- **Flann**: 最近邻搜索

### 视频处理模块组
- **Videoio**: 视频输入输出
- **Video**: 视频分析

### 高级功能模块组
- **Photo**: 计算摄影
- **Stitching**: 图像拼接
- **DNN**: 深度学习
- **GAPI**: 图计算框架
- **ML**: 机器学习

## 最小依赖配置

### 仅图像处理
```
Core → Imgcodecs → Imgproc → Highgui
```

### 特征检测和匹配
```
Core → Imgcodecs → Imgproc → Features2d → Flann
```

### 目标检测
```
Core → Imgcodecs → Imgproc → Objdetect
```

### 深度学习
```
Core → Imgcodecs → Imgproc → DNN
```

### 完整功能
```
所有模块（按依赖关系链接）
```

## 编译依赖说明

### 必需依赖
- **Core**: 所有模块的基础，必须编译
- **Imgcodecs**: 图像读写必需
- **Imgproc**: 大部分高级功能的基础

### 可选依赖
- **DNN**: 需要 protobuf、flatbuffers
- **Videoio**: 需要 FFmpeg（可选）
- **Highgui**: 需要 GUI 库（GTK、Qt 等，可选）

### 第三方库依赖
- **图像格式**: libjpeg, libpng, libtiff, libwebp 等
- **深度学习**: protobuf, flatbuffers
- **视频**: FFmpeg（可选）
- **并行**: TBB, OpenMP（可选）

## 模块独立性

### 完全独立模块
- **Core**: 不依赖任何其他 OpenCV 模块
- **ML**: 仅依赖 Core

### 半独立模块
- **Imgcodecs**: 仅依赖 Core
- **Flann**: 仅依赖 Core

### 依赖链较长的模块
- **Stitching**: Core → Imgproc → Features2d → Calib3d → Stitching
- **GAPI**: Core → Imgproc → DNN → GAPI

## 使用建议

### 轻量级应用
如果只需要基本的图像处理功能，可以只编译：
- Core
- Imgcodecs
- Imgproc
- Highgui（如果需要显示）

### 完整功能应用
如果需要完整的计算机视觉功能，建议编译所有模块。

### 深度学习应用
如果主要使用深度学习功能，重点编译：
- Core
- Imgcodecs
- Imgproc
- DNN

### 嵌入式应用
对于资源受限的嵌入式系统，可以：
- 禁用不需要的模块
- 禁用 GUI 相关功能
- 禁用视频编解码（如果不需要）
- 仅启用必需的图像格式支持

---

*本文档描述了 OpenCV 4.13.0 各模块之间的依赖关系*
