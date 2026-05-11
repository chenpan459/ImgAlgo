# OpenCV 4.13.0 — `hal/ndsrvp` 代码结构分析

本文档说明 **`opencv-4.13.0/hal/ndsrvp`**：**Andes 工具链 DSP / RVP 类指令扩展**（通过 **`nds_intrinsic.h`** 与大量 **`__nds__*` 内建函数**）实现的 **RISC-V 专用 HAL**，将 **`cv_hal_*` 映射到手写向量化与外覆层实现。与 **`hal/riscv-rvv`（RVV）** 在根 CMake 中互斥注册（见下文）。

---

## 1. 定位与启用条件

| 项目 | 说明 |
|------|------|
| **软硬件取向** | 面向 **Andes RISC-V** 及 **`-mext-dsp`**（DSP 扩展）编译环境；源码依赖 **`<nds_intrinsic.h>`**（**`__nds__v_ukadd8`**、**`__nds__zunpkd810`** 等）。 |
| **CMake 选项** | 根目录 **`OCV_OPTION(WITH_NDSRVP "Use Andes RVP extension" …)`**，**`VISIBLE_IF RISCV`**（仅在 **RISC-V** 平台选项中可见；默认随 **`NOT CV_DISABLE_OPTIMIZATION`**）。 |
| **插入 HAL 列表** | **`WITH_NDSRVP`** 开启时 debug 输出 “**Andes RVP 3rdparty NDSRVP enabled**”，并把 **`ndsrvp`** **prepend** 到 **`OpenCV_HAL`**。 |
| **真正编译 `hal/ndsrvp` 的条件**（根 **`CMakeLists.txt`** **`hal STREQUAL "ndsrvp"`** 分支）：**同时具备**<br>① **`CMAKE_C_FLAGS`** 与 **`CMAKE_CXX_FLAGS`** 均 **包含子串 `-mext-dsp`**（Andes GNU 工具链 DSP 开关）；<br>② **`CPU_BASELINE_FINAL`** **不包含** **`RVV`**。<br>否则打印 **`NDSRVP: Andes GNU Toolchain DSP extension is not enabled, disabling ndsrvp...`**，**不** `add_subdirectory`。 |

因此：**同一配置若启用 **RVV** 基线，则不会注册本 HAL**；与 **`hal/riscv-rvv`** 形成清晰分工。

---

## 2. 目录与构建

```
hal/ndsrvp/
├── CMakeLists.txt
├── ndsrvp_hal.hpp          # 总入口：nds_intrinsic.h + opencv2/core/hal/interface.h + include/*.hpp
├── include/
│   ├── core.hpp            # 模板化逐元素算子 + cv_hal_* 宏（体量最大）
│   ├── imgproc.hpp         # imgproc HAL 声明与 cv_hal_* 宏
│   └── features2d.hpp      # 当前为空壳（仅 include guard）
└── src/
    ├── cvutils.hpp / cvutils.cpp   # 内存、边界、打包/解包、向量 clip 等（大量 __nds__ 辅助）
    ├── integral.cpp
    ├── warpAffine.cpp / warpPerspective.cpp
    ├── remap.cpp
    ├── threshold.cpp
    ├── filter.cpp          # cvhalFilter2D：filterInit / filter / filterFree
    ├── medianBlur.cpp
    └── bilateralFilter.cpp
```

**`CMakeLists.txt` 要点**：

- **`file(GLOB … include/*.hpp`、`src/*.cpp`)**，**`add_library(ndsrvp_hal STATIC)`**，**`target_sources(PRIVATE ${headers} ${sources})`**（头文件进目标，便于 IDE/依赖扫描；链接仍以 **`.cpp`** 编译为主）。
- **包含目录**：**`CMAKE_CURRENT_SOURCE_DIR`**（使 **`#include "include/core.hpp"`** 等解析正确）、**`modules/core/include`**、**`modules/imgproc/include`**、**`modules/features2d/include`**。
- **导出变量**：**`NDSRVP_HAL_VERSION`** = **`0.0.1`**，**`NDSRVP_HAL_LIBRARIES`** = **`ndsrvp_hal`**，**`NDSRVP_HAL_HEADERS`** = **`ndsrvp_hal.hpp`**（单头注册），**`NDSRVP_HAL_INCLUDE_DIRS`** = **本目录**。

**无第三方静态库依赖**：与 FastCV/IPP 不同，**不** `target_link_libraries` 外部 SDK。

---

## 3. 总入口 `ndsrvp_hal.hpp`

按顺序包含：

1. **`<nds_intrinsic.h>`**  
2. **`<opencv2/core/hal/interface.h>`**  
3. **`include/core.hpp`**、**`include/imgproc.hpp`**、**`include/features2d.hpp`**

因此 **所有 `#undef cv_hal_* / #define`** 分散在 **core.hpp / imgproc.hpp** 中完成聚合。

---

## 4. `include/core.hpp`：Core HAL 模式

- **命名空间** **`cv::ndsrvp`**。
- **泛型循环**：**`elemwise_binop`**、**`elemwise_unop`** —— 按 **`nlane`** 宽度做向量主循环，尾部标量收尾；**`step`** 会除以 **`sizeof` 元素** 转为“元素步长”。
- **运算符结构体**：如 **`operators_add_t`**、**sub/max/min/absdiff**、**bitwise（and/or/xor/not）**、**cmp** 等，内部调用 **`__nds__v_*` / `__nds__*`** 标量/向量两套实现。
- **`#define cv_hal_add8u`** 等形式为 **函数指针式宏**：直接把 **`cv_hal_add8u`** 指到实例化的 **`elemwise_binop<...>`** 模板，而非单独 **`extern "C"` 函数**（与 IPP/FastCV 的 **`int xxx(...)`** 风格不同）。
- **注释掉的 split/merge**：模板保留在文件中但 **`#undef` 已注释**，当前 **未** 接入 HAL。

---

## 5. `include/imgproc.hpp` 与 `src/*.cpp`

**`imgproc.hpp`** 声明 **`cv::ndsrvp`** 下的 **imgproc 相关入口** 并 **`#define` 到 `cv_hal_*`**，主要包括：

| 类别 | HAL 宏 / 函数 | 实现文件（典型） |
|------|----------------|------------------|
| 积分图 | **`cv_hal_integral`** | **`integral.cpp`** |
| Warp 块行 | **`cv_hal_warpAffineBlockline*`**、**`cv_hal_warpPerspectiveBlockline*`** | **`warpAffine.cpp`**、**`warpPerspective.cpp`** |
| Remap | **`cv_hal_remap32f`** | **`remap.cpp`** |
| 阈值 | **`cv_hal_threshold`** | **`threshold.cpp`** |
| 通用 2D 相关卷积 | **`cv_hal_filterInit`** / **`filter`** / **`filterFree`** | **`filter.cpp`**（内部 **`FilterData`**、核预处理 **`preprocess2DKernel`**） |
| 中值 / 双边 | **`cv_hal_medianBlur`**、**`cv_hal_bilateralFilter`** | **`medianBlur.cpp`**、**`bilateralFilter.cpp`** |

**`include/features2d.hpp`**：仅预留，**无** 宏替换。

---

## 6. `src/cvutils.*` 辅助层

**`cvutils.hpp`**：在 **`nds_intrinsic`** 与 **OpenCV HAL 接口**之上提供 **NDSRVP 内部**工具 —— **`fastMalloc`/`fastFree`**、**`borderInterpolate`（标量+向量）**、**`ndsrvp_u8_u16_expand8`** 等 **pack/unpack** 与 **clip/vclip**，以及错误码 **`ndsrvp_error`**、**`ndsrvp_assert`**。供 **`filter.cpp`** 等源文件包含使用。

---

## 7. 与 **`riscv-rvv` HAL** 的关系（摘要）

| 对比项 | **ndsrvp** | **rvvhal**（`hal/riscv-rvv`） |
|--------|------------|-------------------------------|
| **指令集** | **Andes DSP/RVP（`-mext-dsp`）** | **标准 RISC-V **V**（**`CPU_BASELINE_FINAL` 含 RVV）** |
| **根 CMake 注册条件** | 需 **`-mext-dsp`** 且 **baseline 无 RVV** | 需 **baseline 含 RVV** |

二者在同一 **OpenCV_HAL** 列表里一般不会同时通过门控；以 CMake 配置与 **`getBuildInformation()`** 为准。

---

## 8. 推荐阅读顺序

1. 根 **`CMakeLists.txt`**：**`WITH_NDSRVP`**、**`ndsrvp` 分支**的 **`CMAKE_*_FLAGS`** 与 **RVV** 判断。  
2. **`ndsrvp_hal.hpp`** → **`include/core.hpp`**（理解模板 + **`__nds__`** 映射范围）。  
3. **`include/imgproc.hpp`** → 对应 **`src/*.cpp`**。  
4. **`src/cvutils.hpp`**：**内部向量与边界工具**。

---

## 9. 路径与版本说明

- 分析对象：`/home/work2/ImgAlgo/opencv-4.13.0/hal/ndsrvp`。  
- 内建函数与可用类型以 **所用 Andes 工具链 / `nds_intrinsic.h`** 为准；**`features2d`** 若在未来版本补充，以当前树 **`include/features2d.hpp`** 为准。

---

*文档用于源码导航；NDSRVP 与 OpenCV 版本演进可能增减 HAL 入口或编译条件，以当前树为准。*
