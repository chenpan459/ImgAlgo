# OpenCV 4.13.0 — `hal/ipp` 代码结构分析

本文档说明 **`opencv-4.13.0/hal/ipp`**：在 **Intel Integrated Performance Primitives（IPP）** 之上实现的 **OpenCV HAL** 适配层。目标平台主要为 **x86 / x86_64**（与根 **`CMakeLists.txt`** 中 **`WITH_IPP`** 等选项一致）。通过 **`#undef` / `#define` 将 `cv_hal_*` 映射到 `ipp_hal_*`**，在不支持的配置下返回 **`CV_HAL_ERROR_NOT_IMPLEMENTED`**，由 OpenCV 核心回退实现。

---

## 1. 定位与依赖

| 项目 | 说明 |
|------|------|
| **上游** | **IPP** 头文件与库（**`IPP_INCLUDE_DIRS`**、**`IPP_LIBRARIES`**），以及可选 **IPP IW**（**`IPP_IW_LIBRARY`**、**`HAVE_IPP_IW`**）。**ICV 内置 IPP** 时使用 **`HAVE_IPP_ICV`** 与 **`ippicv.h`**（或旧版 **`ipp.h`**），详见 **`include/ipp_utils.hpp`**。 |
| **根 CMake** | **`WITH_IPP`**（默认在 **非 MINGW 且未禁用优化** 时开启）触发探测；**`HAVE_IPP`** 为真时把 **`ipp`** **prepend** 到 **`OpenCV_HAL`**，并 **`add_subdirectory(hal/ipp)`**，**`ocv_hal_register(IPP_HAL_*)`**。 |
| **本目录产物** | 静态库 **`ipphal`**（工程名 **`ipphal`**），链接 **`${IPP_IW_LIBRARY} ${IPP_LIBRARIES`**。 |
| **与 core 中 IPP 的关系** | **`hal/ipp/CMakeLists.txt`** 注释说明：**`HAVE_IPP_ICV` / `HAVE_IPP_IW`** 暂以 **PRIVATE** 宏传入，避免与 OpenCV 自带 IPP 集成重复定义；未来若 IPP 完全迁到 HAL，可改为 **PUBLIC**。 |

---

## 2. 目录结构

```
hal/ipp/
├── CMakeLists.txt
├── include/
│   ├── ipp_utils.hpp        # IPP_VERSION_X100、头文件选择（ICV vs 系统 IPP）、CV_INSTRUMENT_FUN_IPP
│   ├── ipp_hal_core.hpp     # 覆盖 cv_hal_*（core）
│   └── ipp_hal_imgproc.hpp  # 覆盖 cv_hal_*（imgproc：warp/remap，受 IPP 版本与 IW 条件约束）
└── src/
    ├── precomp_ipp.hpp      # IppiSize、深度/插值/边界枚举转换；可选 IW；线程数启发（ippiSuggestThreadsNum 等）
    ├── mean_ipp.cpp         # meanStdDev（IPP ≥ 7.0 段）
    ├── minmax_ipp.cpp       # minMaxIdx（带 mask step）
    ├── norm_ipp.cpp         # norm / normDiff（含按 IPP 版本关闭部分路径的宏）
    ├── sum_ipp.cpp          # sum（IPP ≥ 7.0 段）
    ├── cart_polar_ipp.cpp   # polarToCart 32f/64f（IPP 核心，不依赖 ipp_hal 中大段 #if）
    ├── transforms_ipp.cpp   # transpose2d；ifdef HAVE_IPP_IW 时含 flip 等 IW 路径
    └── warp_ipp.cpp         # warpAffine / warpPerspective / remap32f（文件级要求 IPP ≥ 8.1 ABI）
```

**`ocv_hal_register` 登记的对外头文件**：仅 **`ipp_hal_core.hpp`**、**`ipp_hal_imgproc.hpp`**。**`ipp_utils.hpp`** 由二者包含，不单独导出。

---

## 3. 构建要点（`hal/ipp/CMakeLists.txt`）

- **版本元数据**：**`IPP_HAL_VERSION`** = **`0.0.1`**，**`IPP_HAL_LIBRARIES`** = **`ipphal`**，**`IPP_HAL_INCLUDE_DIRS`** = 本目录 **`include`**。
- **源文件列表**：**显式列举** 7 个 **`src/*_ipp.cpp`**（非 `GLOB`）。
- **可选宏**：**`HAVE_IPP_ICV`**、**`HAVE_IPP_IW`** → **PRIVATE** **`target_compile_definitions`**；**`WITH_IPP_CALLS_ENFORCED`** → **`IPP_CALLS_ENFORCED`**，用于在 **`warp_ipp.cpp`** 等中放宽/强制走 IPP 的配置表（与 OpenCV 默认严格对齐时的 **`impl` 表**不同）。
- **包含目录**：**`include`**、**`src`**、**`modules/core/include`**、**`modules/imgproc/include`**、**`${IPP_INCLUDE_DIRS}`**。
- **链接**：**`PUBLIC ${IPP_IW_LIBRARY} ${IPP_LIBRARIES}`**，保证 IW 与 IPP 符号对 **`ipphal`** 使用者可见。
- **安装**：**`BUILD_SHARED_LIBS=OFF`** 时安装 **`ipphal`** 归档到第三方库安装路径。

---

## 4. 版本与条件编译（`ipp_utils.hpp` / 各 `.hpp`）

- **`IPP_VERSION_X100`**：由 **`ippversion.h`** 的 **major / minor / update** 合成，用于 **#if** 裁剪 API。
- **Core（`ipp_hal_core.hpp`）**  
  - **`IPP_VERSION_X100 >= 700`**：**`meanStdDev`**、**`minMaxIdxMaskStep`**、**`norm`**、**`normDiff`**、**`sum`**。  
  - 部分 **norm** 路径在特定 IPP 版本被 **`IPP_DISABLE_NORM_8U`**、**`IPP_DISABLE_NORM_INF_16U_C1MR`** 关闭（注释标明与精度或崩溃测试相关）。  
  - **`HAVE_IPP_IW`**：**`flip`**。  
  - **无版本 guard**：**`polarToCart32f/64f`**、**`transpose2d`**（实现仍在对应 **`.cpp`** 内做能力判断）。
- **Imgproc（`ipp_hal_imgproc.hpp`）**  
  - **`IPP_VERSION_X100 >= 810`**：整文件内声明 **warp / remap** 类 HAL。  
  - **`HAVE_IPP_IW`**：**`warpAffine`**。  
  - 始终（在同一 **#if IPP>=810** 块内）：**`warpPerspective`**、**`remap32f`**。

上述与 **`warp_ipp.cpp`** 文件头 **`#if IPP_VERSION_X100 >= 810`** 一致：**仿射/透视/remap** 依赖较新 IPP 集成接口。

---

## 5. 实现文件职责摘要

| 文件 | 主要职责 |
|------|----------|
| **`mean_ipp.cpp`** | **`ipp_hal_meanStdDev`**：按类型/通道/Mask 选择 **`ippiMean_*`** 等，**`CV_INSTRUMENT_FUN_IPP`** 包裹调用。 |
| **`minmax_ipp.cpp`** | **`ipp_hal_minMaxIdxMaskStep`** 等极值与索引。 |
| **`norm_ipp.cpp`** | **`ipp_hal_norm`** / **`ipp_hal_normDiff`**，含按版本禁用的路径。 |
| **`sum_ipp.cpp`** | **`ipp_hal_sum`**，多类型 **Hint / Non-Hint** **`ippiSum_*`**。 |
| **`cart_polar_ipp.cpp`** | **`ipp_hal_polarToCart*`**：调用 **`ippsPolarToCart_*`**；拒绝 **in-place** 与 **角度为度** 的情况。 |
| **`transforms_ipp.cpp`** | **`ipp_hal_transpose2d`**（**`ippiTranspose_*`**）；**`#ifdef HAVE_IPP_IW`** 分支下 **flip** 等与 **IwiImage** 相关逻辑。 |
| **`warp_ipp.cpp`** | **`ipp_hal_warpAffine` / `warpPerspective` / `remap32f`**：IW **`iwiWarpAffine`** 等、分块并行 **`ParallelLoopBody`**、与 **`ippiGetInterpolation` / `ippiGetBorderType`** 配合；含 **`IPP_CALLS_ENFORCED`** 与默认 **`impl` 能力表** 两套行为。 |

**`precomp_ipp.hpp`**：集中 **OpenCV ↔ IPP** 的类型与枚举映射，以及基于 **L2 cache** 与图像尺寸的 **线程划分** 辅助函数；**`HAVE_IPP_IW`** 时增加 **`IwiImage`** 重载。

---

## 6. 与根 **`OpenCV_HAL` 顺序**

若同时启用多种 HAL，**`ipp`** 会在 **`HAVE_IPP`** 时被插到列表前部（与 **FastCV、Carotene** 等 prepend 逻辑类似）。最终是否命中 IPP 仍取决于各 **`ipp_hal_*`** 返回值及各模块调用链。

---

## 7. 推荐阅读顺序

1. 根 **`CMakeLists.txt`**：**`WITH_IPP`**、**`HAVE_IPP`**、**`foreach(hal … ipp)`** 分支与 **status** 输出。  
2. **`cmake`** 中 **IPP / IPPICV / IPP IW** 探测脚本（与 **`IPP_ROOT_DIR`**、**`BUILD_WITH_DYNAMIC_IPP`** 等相关）。  
3. **`include/ipp_hal_core.hpp`**、**`ipp_hal_imgproc.hpp`**：完整 **`cv_hal_*` → `ipp_hal_*`** 映射与版本宏。  
4. 按需阅读 **`src/*_ipp.cpp`** 与 **`precomp_ipp.hpp`**。

---

## 8. 路径与版本说明

- 分析对象：`/home/work2/ImgAlgo/opencv-4.13.0/hal/ipp`。  
- 实际启用与 **IPP 发行版号**、**ICV / 独立 IPP**、**IW 是否参与编译** 强相关；以目标环境的 **`getBuildInformation()`** 与 CMake 摘要为准。

---

*文档用于源码导航；IPP 与 OpenCV 版本演进可能调整宏与接口封装，以当前树为准。*
