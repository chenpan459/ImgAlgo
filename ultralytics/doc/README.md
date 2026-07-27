# Ultralytics YOLO 源码分析文档

本目录为对仓库中 `ultralytics-8.4.107`（`ultralytics.__version__ = "8.4.107"`）的源码分析文档，面向阅读、二次开发与调试。

## 文档索引

| 文档 | 内容 |
|------|------|
| [01_系统概述.md](01_系统概述.md) | 项目简介、版本特性、依赖、目录结构 |
| [02_系统架构.md](02_系统架构.md) | 分层架构、调用链、task_map 设计 |
| [03_核心模块详解.md](03_核心模块详解.md) | Engine / Models / NN / Data / Cfg |
| [04_算法流水线.md](04_算法流水线.md) | Train / Val / Predict / Export / Track |
| [05_任务与网络.md](05_任务与网络.md) | 检测/分割/姿态等任务、YOLO26、parse_model |
| [06_配置与使用.md](06_配置与使用.md) | default.yaml、CLI、Python API、导出 |
| [07_源码文件索引.md](07_源码文件索引.md) | 类/文件对照与阅读顺序 |

## 源码根路径

```text
ultralytics/ultralytics-8.4.107/
```

## 快速结论

- Ultralytics 是面向 YOLO 家族的统一 Python 包：**训练 / 验证 / 预测 / 跟踪 / 导出 / 基准测试**。
- 默认模型族为 **YOLO26**（如 `yolo26n.pt`），任务含 detect / segment / classify / pose / obb / **semantic** / **depth**。
- 核心设计：`YOLO`/`Model` 门面通过 **`task_map`** 懒加载任务专属 `Trainer` / `Validator` / `Predictor` / `nn.TaskModel`。
- 另支持 RT-DETR、SAM/SAM2/SAM3、FastSAM、NAS、YOLO-World、YOLOE 等架构。
- CLI 入口：`yolo` / `ultralytics` → `ultralytics.cfg:entrypoint`。

## 官方文档

- 在线文档：https://docs.ultralytics.com
- 商业许可：https://www.ultralytics.com/license
- 本仓库许可证：**AGPL-3.0**
