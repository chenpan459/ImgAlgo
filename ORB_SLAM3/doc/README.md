# ORB-SLAM3 源码分析文档

本目录为对仓库中 `ORB_SLAM3-1.0-release`（V1.0, 2021-12-22）的源码分析文档，面向阅读、二次开发与调试。

## 文档索引

| 文档 | 内容 |
|------|------|
| [01_系统概述.md](01_系统概述.md) | 项目简介、特性、依赖、目录结构、版本要点 |
| [02_系统架构.md](02_系统架构.md) | 线程模型、System 入口、数据流、Atlas 多地图 |
| [03_核心模块详解.md](03_核心模块详解.md) | Tracking / LocalMapping / LoopClosing / 特征 / 相机模型 |
| [04_算法流水线.md](04_算法流水线.md) | 初始化、跟踪、建图、回环、地图融合流程 |
| [05_IMU与优化.md](05_IMU与优化.md) | IMU 预积分与初始化、g2o 因子、Optimizer API |
| [06_配置与使用.md](06_配置与使用.md) | YAML 配置、Examples、地图序列化、评估 |
| [07_源码文件索引.md](07_源码文件索引.md) | 类/文件对照表与关键 API |

## 源码根路径

```text
ORB_SLAM3/ORB_SLAM3-1.0-release/
```

## 快速结论

- ORB-SLAM3 是支持 **纯视觉 / 视觉惯性 / 多地图** 的实时 SLAM 库。
- 传感器：单目、双目、RGB-D，以及对应的 IMU 组合。
- 相机模型：针孔（PinHole）与鱼眼（KannalaBrandt8）。
- 主线程跑 Tracking；LocalMapping、LoopClosing、Viewer 为独立线程。
- 多地图由 `Atlas` 管理；丢失后可新建地图，重访时通过 Place Recognition 做 **Loop** 或 **Merge**。

## 相关论文

1. Campos et al., **ORB-SLAM3**, IEEE TRO 2021.
2. Campos et al., **Inertial-Only Optimization for Visual-Inertial Initialization**, ICRA 2020.
3. Elvira et al., **ORBSLAM-Atlas**, IROS 2019.
4. Mur-Artal & Tardós, **ORB-SLAM2**, IEEE TRO 2017.
5. Mur-Artal et al., **ORB-SLAM**, IEEE TRO 2015.
