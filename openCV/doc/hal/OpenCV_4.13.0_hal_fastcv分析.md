# OpenCV 4.13.0 — `hal/fastcv` 代码结构分析

本文档说明 **`opencv-4.13.0/hal/fastcv`**：在 **Qualcomm FastCV** 预编译库之上实现的 **OpenCV HAL 封装层**，通过 **`#define` 覆盖 `cv_hal_*` 符号**，让 **core / imgproc** 在支持的平台上优先走 FastCV 实现；不满足条件时返回 **`CV_HAL_ERROR_NOT_IMPLEMENTED`**，由 OpenCV 回退到默认实现。

---

## 1. 定位与依赖关系

| 项目 | 说明 |
|------|------|
| **上游库** | **FastCV**（`libfastcv.a` + `fastcv.h` 等头文件），由 **CMake** 探测：用户指定 **`FastCV_INCLUDE_PATH` / `FastCV_LIB_PATH`**，或通过 **`3rdparty/fastcv/fastcv.cmake`** 从 **`opencv_3rdparty`** 下载预编译包。 |
| **CMake 开关** | 根 **`CMakeLists.txt`** 中 **`OCV_OPTION(WITH_FASTCV … OFF)`**（默认关闭）。打开后 **`cmake/OpenCVFindLibsPerf.cmake`** 设置 **`HAVE_FASTCV`** 并配置 **`FASTCV_LIBRARY`（常为 IMPORTED 目标 `fastcv`，链接 `dl`）**。Android **打包配置**里可见 **`WITH_FASTCV=ON`**（见 **`platforms/android/fastcv.config.py`**）。 |
| **本目录产物** | 静态库 **`fastcv_hal`**（`src/*.cpp`），**`target_link_libraries(fastcv_hal PUBLIC ${FASTCV_LIBRARY})`**。 |
| **版权** | 头文件与实现为 **Qualcomm Innovation Center, Inc.**，**Apache-2.0**。 |

**FastCV 预编译包平台**（见 **`3rdparty/fastcv/fastcv.cmake`**，包名带日期标签如 **`2025_07_09`**）：

- **Android**：**aarch64**、**armv7**
- **Linux（非 Apple / 非 iOS / 非 visionOS）**：仅 **aarch64**；**32 位 Linux arm** 当前脚本打印不支持并可能无法设置包名

根目录 **`CMakeLists.txt`** 中，仅当 **`HAVE_FASTCV`** 为真时把 **`fastcv`** prepend 到 **`OpenCV_HAL`**；随后在 **`foreach(hal ${OpenCV_HAL})`** 里，仅当满足下方平台条件时才 **`add_subdirectory(hal/fastcv)`** 并 **`ocv_hal_register(FASTCV_HAL_*)`**：

**`(ARM OR AARCH64) AND (ANDROID OR (UNIX AND NOT APPLE AND NOT IOS AND NOT XROS))`**

不满足则打印 **`FastCV: fastcv is not available, disabling fastcv...`**。

---

## 2. 目录结构

```
hal/fastcv/
├── CMakeLists.txt
├── include/
│   ├── fastcv_hal_core.hpp    # 覆盖 cv_hal_*（core）：LUT、Hamming、乘/转置、meanStdDev、flip、rotate、addWeighted、mul、SVD、gemm 等
│   ├── fastcv_hal_imgproc.hpp # 覆盖 cv_hal_*（imgproc）：medianBlur、sobel、boxFilter、adaptiveThreshold、gaussianBlurBinomial、warpPerspective、pyrdown、颜色转换、Canny 等
│   └── fastcv_hal_utils.hpp   # FastCV 初始化、错误码映射、调试宏（不列入 FASTCV_HAL_HEADERS，仅供 .cpp 使用）
└── src/
    ├── fastcv_hal_core.cpp
    ├── fastcv_hal_imgproc.cpp
    └── fastcv_hal_utils.cpp   # getFastCVErrorString、border/interpolation 字符串等
```

**`CMakeLists.txt` 要点**（在 **`HAVE_FASTCV`** 为真时）：

- **`FASTCV_HAL_VERSION`**：`0.0.1`
- **`FASTCV_HAL_LIBRARIES`**：`fastcv_hal`
- **`FASTCV_HAL_INCLUDE_DIRS`**：本目录 **`include`**
- **`FASTCV_HAL_HEADERS`**：仅 **`fastcv_hal_core.hpp`**、**`fastcv_hal_imgproc.hpp`**（与 **`ocv_hal_register`** 一致）
- 源文件：**`file(GLOB … src/*.cpp)`**，包含 **utils** 与 **core/imgproc** 实现
- **include**：**`modules/core/include`**、**`modules/imgproc/include`**、本目录 include、**`FastCV_INCLUDE_PATH`**

---

## 3. HAL 绑定方式

与多数内置 HAL 相同：在头文件中 **`#undef cv_hal_xxx` / `#define cv_hal_xxx fastcv_hal_xxx`**，将 OpenCV 内部的 **`cv::hal`** 调用解析到 **`fastcv_hal_*`**。

**Core 侧覆盖**（摘自 **`fastcv_hal_core.hpp`**，节选）：

- **`cv_hal_lut`**、**`cv_hal_normHammingDiff8u`**、**`cv_hal_mul8u16u`**、**`cv_hal_sub8u32f`**、**`cv_hal_transpose2d`**
- **`cv_hal_meanStdDev`**、**`cv_hal_flip`**、**`cv_hal_rotate90`** → **`fastcv_hal_rotate`**
- **`cv_hal_addWeighted8u`**、**`cv_hal_mul8u` / `mul16s` / `mul32f`**
- **`cv_hal_SVD32f`**、**`cv_hal_gemm32f`**

**Imgproc 侧覆盖**（摘自 **`fastcv_hal_imgproc.hpp`**，节选）：

- **`cv_hal_medianBlur`**、**`cv_hal_sobel`**、**`cv_hal_boxFilter`**、**`cv_hal_adaptiveThreshold`**
- **`cv_hal_gaussianBlurBinomial`**、**`cv_hal_warpPerspective`**、**`cv_hal_pyrdown`**
- **`cv_hal_cvtBGRtoHSV`**、**`cv_hal_cvtBGRtoYUVApprox`**、**`cv_hal_canny`**

具体是否真正走 FastCV，由各 **`fastcv_hal_*`** 函数内部的 **类型、尺寸、核大小、scale/delta、in-place** 等校验决定；不通过则 **`CV_HAL_RETURN_NOT_IMPLEMENTED`**。

---

## 4. 运行时与工具宏（`fastcv_hal_utils.hpp`）

- **`FastCvContext`**：单例，首次使用在构造里调用 **`fcvSetOperationMode(FASTCV_OP_CPU_PERFORMANCE)`**；失败则 **`isInitialized = false`**，后续 **`INITIALIZATION_CHECK`** 直接返回 **`CV_HAL_ERROR_UNKNOWN`**。
- **`CV_HAL_RETURN`**：将 **`fcvStatus`** 映射为 **`CV_HAL_ERROR_OK`** / **`CV_HAL_ERROR_NOT_IMPLEMENTED`**（参数不支持、硬件未就绪等）/ **`CV_HAL_ERROR_UNKNOWN`**，并打日志。
- **`CV_HAL_RETURN_NOT_IMPLEMENTED`**：显式回退 OpenCV 默认路径。
- **`fcv.h`** 由 **`#include "fastcv.h"`** 引入（来自 **`FastCV_INCLUDE_PATH`**）。

---

## 5. 实现特点（阅读源码时的提示）

- **大量条件限制**：例如 **`fastcv_hal_lut`** 在较小分辨率上直接 **`NOT_IMPLEMENTED`**；**flip** 对部分 **`flip_mode`** 或过小分辨率回退；**sobel** 限制 **dx/dy 阶数、scale/delta、dst 深度、ksize** 等。
- **并行**：部分算子使用 **`cv::parallel_for_`** 与自定义 **`cv::ParallelLoopBody`**（如 LUT 按行、Sobel 分块、**`mul*`** 按条带）。
- **In-place**：多处显式拒绝 **`src_data == dst_data`**。
- **`fastcv_hal_core.cpp`** 中 **SVD/gemm** 等会在 FastCV 与 OpenCV 数据结构之间做适配（需读剩余行以跟踪完整分支）。

---

## 6. 与 **`OpenCV_HAL` 列表顺序**

根 **`CMakeLists.txt`** 在 **IPP、OpenVX、FastCV、KleidiCV、Carotene…** 等之间按条件 **prepend** 各 HAL。**实际优先级**还受 **`ocv_hal_register` 注册顺序**与各模块 **`cv::hal`** 实现中的 **尝试顺序** 影响；部署时以 **`getBuildInformation()`** 与单算子 profiling 为准。

---

## 7. 推荐阅读顺序

1. **`cmake/OpenCVFindLibsPerf.cmake`** 中 **FastCV** 段与 **`3rdparty/fastcv/fastcv.cmake`**（如何得到 **`HAVE_FASTCV`**）。
2. 根 **`CMakeLists.txt`** 中 **`hal STREQUAL "fastcv"`** 分支（平台门控）。
3. **`hal/fastcv/include/fastcv_hal_*.hpp`**：掌握了哪些 **`cv_hal_*`** 被替换。
4. **`hal/fastcv/src/fastcv_hal_core.cpp` / `fastcv_hal_imgproc.cpp`**：参数校验与 **`fcv*`** 调用。
5. **`hal/fastcv/CMakeLists.txt`**：与 **`fastcv`** IMPORTED 库的链接关系。

---

## 8. 版本与路径说明

- 分析对象：`/home/work2/ImgAlgo/opencv-4.13.0/hal/fastcv`。
- 第三方包 **commit / 文件名** 以仓库内 **`3rdparty/fastcv/fastcv.cmake`** 为准；升级 OpenCV 时可能变更。

---

*文档用于源码导航；FastCV 与 OpenCV 版本演进可能调整覆盖的 HAL 入口与条件判断，以当前树为准。*
