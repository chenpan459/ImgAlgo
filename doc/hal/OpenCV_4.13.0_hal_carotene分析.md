# OpenCV 4.13.0 — `hal/carotene` 代码结构分析

本文档说明 **`opencv-4.13.0/hal/carotene`**：**Carotene** 是一套面向 **ARM NEON** 的底层图像/数值算子库，经 **`hal/carotene/hal/`** 与 OpenCV 的 **`tegra_hal`**（历史命名，NVIDIA Tegra/嵌入式场景）对接为 **内置 HAL**，供 **`cv::hal`** / **`imgproc`** 等路径在支持平台上**替换默认实现**。

---

## 1. 定位与简述

- **`README.md`**：*Carotene is a low-level library containing optimized CPU routines that are useful for computer vision algorithms.*
- **平台取向**：编译上依赖 **`WITH_NEON`**（在 **GCC/Clang** 下通过 **`CPU_BASELINE_FINAL` 含 NEON** 等条件在父 **`hal/CMakeLists.txt`** 中启用，参见仓库根 **`CMakeLists.txt`** 中 **`hal STREQUAL "carotene"`** 分支）。
- **命名空间**：默认可 **`CAROTENE_NS="carotene"`**；在 **HAL 集成层**中常强制为 **`carotene_o4t`**（见 **`hal/CMakeLists.txt`** **`CAROTENE_NS "carotene_o4t" CACHE ... FORCE`**），与 **`tegra_hal.hpp`** 内 **`#define CAROTENE_NS carotene_o4t`** 一致。

---

## 2. 目录结构

```
hal/carotene/
├── CMakeLists.txt          # 子工程：OBJECT 库 carotene_objs → 静态库 carotene
├── README.md
├── include/carotene/
│   ├── definitions.hpp     # 类型、策略、平台能力查询等
│   ├── types.hpp           # Size2D、边界/策略枚举等
│   └── functions.hpp       # 全部 Carotene 例程声明（按数据类型重载，体量很大）
├── src/
│   ├── *.cpp               # 各算子 NEON 实现
│   ├── common.hpp / common.cpp
│   ├── intrinsics.hpp     # 向量 intrinsic 包装
│   └── …                   # 若干 *.hpp 辅助（remap、saturate_cast、separable_filter 等）
└── hal/
    ├── CMakeLists.txt      # OpenCV 注册：编译 Carotene、生成 tegra_hal 静态库
    ├── tegra_hal.hpp       # OpenCV HAL 与 Carotene 的宏/模板桥（大量映射）
    └── dummy.cpp           # Xcode 等构建占位
```

**关于 `impl.cpp`**：在 **`hal/carotene/hal/CMakeLists.txt`**（约 72–81 行）里，**GCC** 下对 **`impl.cpp`** 与 **`$<TARGET_OBJECTS:carotene_objs>`** 调了 **`set_source_files_properties`**；但 **4.13.0 源码树中 `hal/carotene/hal/` 下并无 `impl.cpp`**，属于**悬空引用**（多为历史遗留或笔误），对实际链接无影响。**`tegra_hal`** 的源集仅为 **`$<TARGET_OBJECTS:carotene_objs> dummy.cpp`**（约 86 行）。

---

## 3. 构建要点（`hal/carotene/CMakeLists.txt`）

- **`add_library(carotene_objs OBJECT …)`**：所有 **`src/**/*.cpp|hpp`** 进入 **OBJECT**，再 **`add_library(carotene STATIC … $<TARGET_OBJECTS:carotene_objs> dummy.cpp)`**。
- **`-fvisibility=hidden`**（**GNU**）：减小导出符号。
- **GCC 内联增长参数**（**`<10` vs `≥10`** 分两支）：注释写明对 **matchTemplate、goodFeaturesToTrack、cornerHarris** 等有几到三成量级的收益。
- **`WITH_NEON`**：定义 **`DWITH_NEON`** 传给 Carotene 源码。
- **`MINGW`**：**`_USE_MATH_DEFINES`**

父目录 **`hal/carotene/hal/CMakeLists.txt`** 额外设置 **Tegra 历史编译标志**、`compile_carotene()`、把 **`carotene` 头复制到 build 树**，并 **`PARENT_SCOPE` 导出**：

- **`CAROTENE_HAL_VERSION`**（如 **0.0.1**）
- **`CAROTENE_HAL_LIBRARIES`** → **`tegra_hal`**
- **`CAROTENE_HAL_HEADERS`** → **`carotene/tegra_hal.hpp`**
- **`CAROTENE_HAL_INCLUDE_DIRS`**

供根 **`ocv_hal_register(CAROTENE_HAL_*)`** 使用。

---

## 4. `src/*.cpp` 功能分组（按文件名）

下列按**文件名**归类，便于检索（与 OpenCV **imgproc/core/features2d/video** 等 API 对应，非严格一一映射）。

| 类别 | 代表源文件 |
|------|------------|
| **算术/像素运算** | `add.cpp`、`sub.cpp`、`mul.cpp`、`div.cpp`、`add_weighted.cpp`、`absdiff.cpp`、`bitwise.cpp`、`cmp.cpp`、`min_max.cpp`、`minmaxloc.cpp`、`threshold.cpp`、`in_range.cpp` |
| **规约与统计** | `sum.cpp`、`norm.cpp`、`meanstddev.cpp`、`reduce.cpp`、`count_nonzero.cpp`、`dot_product.cpp`、`phase.cpp`、`magnitude.cpp` |
| **类型转换** | `convert.cpp`、`convert_scale.cpp`、`convert_depth.cpp`、`colorconvert.cpp` |
| **滤波 / 卷积** | `convolution.cpp`、`separable_filter.cpp`、`blur.cpp`、`gaussian_blur.cpp`、`median_filter.cpp`、`sobel.cpp`、`scharr.cpp`、`laplacian.cpp`、`morph.cpp`、`bilateral_filter.cpp` |
| **几何与重映射** | `resize.cpp`、`remap.cpp`、`warp_affine.cpp`、`warp_perspective.cpp`、`pyramid.cpp`、`flip.cpp` |
| **特征与角点** | `fast.cpp`、`template_matching.cpp` |
| **光流** | `opticalflow.cpp` |
| **边缘** | `canny.cpp` |
| **通道** | `channel_extract.cpp`、`channels_combine.cpp` |
| **其它** | `integral.cpp`、`accumulate.cpp`、`fill_minmaxloc.cpp`、`remap.hpp` 等辅助 |

完整列表以 **`file(GLOB_RECURSE … src)`** 结果为准。

---

## 5. HAL 绑定层：`hal/tegra_hal.hpp`

- **版权声明**：原 **NVIDIA** BSD 许可（与 **`functions.hpp`** 头一致）。
- **作用**：**`#include "carotene/functions.hpp"`** 后，通过大量 **宏与模板**（如 **`TegraGenOp_Invoker`**、**`RANGE_DATA`**）把 **OpenCV `Mat` 步长/Range** 转成 **`CAROTENE_NS::Size2D`** 与裸指针调用。
- **与 OpenCV core 集成**：包含 **`opencv2/core/base.hpp`**，并可用 **`cv::ParallelLoopBody`** 做行块并行（宏 **`PARALLEL_CORE`** 默认 **0**，可按需打开）。

阅读 **“某 OpenCV 算子是否走 Carotene”** 时，可在 **`tegra_hal.hpp`** 内搜索对应 **HAL 函数名** 或 **`CAROTENE_NS::`** 调用。

---

## 6. 与根工程 CMake 的关系

在 **根 `opencv-4.13.0/CMakeLists.txt`** 中，仅当 **`OpenCV_HAL`** 列表包含 **`carotene`** 且 **`CPU_BASELINE_FINAL` 匹配 NEON** 时 **`add_subdirectory(hal/carotene/hal)`**（不是直接 **`hal/carotene`**），从而注册 **Carotene HAL**；否则打印 **“Carotene: NEON is not available, disabling carotene...”**。

---

## 7. 推荐阅读顺序

1. **`include/carotene/functions.hpp`**：`isSupportedConfiguration` 与各算子原型。  
2. **按需阅读对应 `src/<name>.cpp`** 与 **`intrinsics.hpp`**。  
3. **`hal/tegra_hal.hpp`**：从 OpenCV 侧入口追到 **Carotene** 调用。  
4. **`hal/carotene/hal/CMakeLists.txt`**：产物名 **tegra_hal** 与 **`ocv_hal_register` 变量**。  

---

## 8. 版本与路径说明

- 分析对象：`/home/work2/ImgAlgo/opencv-4.13.0/hal/carotene`。  
- **是否编入安装包** 取决于 **`OpenCV_HAL`** 与 **NEON** 探测；以 **CMake 配置摘要** 与 **`getBuildInformation()`** 为准。

---

*文档用于源码导航；Carotene 与 OpenCV 版本演进可能调整源文件列表，以当前树 `src/` 为准。*
