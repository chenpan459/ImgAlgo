# OpenCV 4.13.0 — `hal/riscv-rvv` 代码结构分析

本文档说明 **`opencv-4.13.0/hal/riscv-rvv`**：面向 **RISC-V Vector（RVV）** 向量扩展的 **OpenCV HAL** 实现，使用 **`<riscv_vector.h>`** 与 **`__riscv_*`** 内建函数；静态库目标名 **`rvv_hal`**，注册名为 **`rvvhal`**（根 **`CMakeLists.txt`** 中 **HAL 名称**）。

---

## 1. 定位与启用条件

| 项目 | 说明 |
|------|------|
| **指令集** | 标准 **RISC-V Vector ISA**，依赖编译器预定义 **`__riscv_v`** 与 **`#include <riscv_vector.h>`**（见 **`rvv_hal.hpp`**）。 |
| **CMake 选项** | **`OCV_OPTION(WITH_HAL_RVV "Use HAL RVV optimizations" …)`**，**`VISIBLE_IF RISCV`**（仅 **RISC-V** 交叉/本机构建时可见；默认 **`NOT CV_DISABLE_OPTIMIZATION`**）。 |
| **HAL 列表** | **`WITH_HAL_RVV`** 为真时 **debug** 输出 “**Enable RVV HAL acceleration**”，并将 **`rvvhal`** **prepend** 到 **`OpenCV_HAL`**。 |
| **子目录编译门控**（根 **`foreach(hal …)`** 中 **`hal STREQUAL "rvvhal"`**）：仅当 **`CPU_BASELINE_FINAL`** **包含** **`RVV`** 时 **`add_subdirectory(hal/riscv-rvv)`** 并 **`ocv_hal_register(RVV_HAL_*)`**；否则打印 **`RVV HAL: RVV is not available, disabling RVV HAL...`**。 |

**与 `hal/ndsrvp` 的互斥关系**：**NDSRVP** 要求 **`-mext-dsp`** 且 **baseline 无 RVV**；**RVV HAL** 要求 **baseline 含 RVV**。二者通常不会在同一配置中同时通过门控。

---

## 2. 工具链/版本开关（`rvv_hal.hpp`）

预处理器从 **`__riscv_v`** 数值区分实现档位（节选）：

| 宏 | 条件（摘要） | 作用 |
|----|----------------|------|
| **`CV_HAL_RVV_1P0_ENABLED`** | **`__riscv_v == 1000000`** | **RVV 1.0** 档；**`include/core.hpp`、`imgproc.hpp`、`features2d.hpp`** 中 **绝大部分 `cv_hal_*` 宏** 在该条件下启用。 |
| **`CV_HAL_RVV_071_ENABLED`** | **`__riscv_v == 7000`** 且 **GCC 10.4** 且定义 **`__THEAD_VERSION__`** | **玄铁 / T-Head** 等旧版向量约定；**`imgproc.hpp`** 末尾仅对 **`cvtBGRtoBGR`** 等少量入口做补充 **`#define`**。 |

若二者均未满足，仍可能包含头文件骨架，但 **`riscv_vector.h` 与大量 HAL 替换不会生效**。

---

## 3. 目录与构建

```
hal/riscv-rvv/
├── CMakeLists.txt          # 静态库 rvv_hal；GLOB src/**/*.cpp 与 include/*.hpp
├── rvv_hal.hpp             # __riscv_v 判断 + 包含 types / core / imgproc / features2d
├── include/
│   ├── types.hpp           # RVV LMUL、RVV&lt;T,lmul&gt; 等类型与元编程（riscv_vector.h）
│   ├── core.hpp            # cv::rvv_hal::core — cv_hal_* 声明与 #define
│   ├── imgproc.hpp         # cv::rvv_hal::imgproc
│   └── features2d.hpp      # cv::rvv_hal::features2d — FAST / cv_hal_FASTv2
└── src/
    ├── core/               # merge、norm、lut、SVD、div、compare、dxt、…
    ├── imgproc/            # filter、resize、warp、color、canny、…
    └── features2d/         # fast.cpp
```

**`CMakeLists.txt` 要点**：

- **`file(GLOB … "${RVV_HAL_SOURCE_DIR}/**/*.cpp")`** 递归收集 **`src`** 下所有 **`.cpp`**。  
- **`add_library(rvv_hal STATIC)`**，**`target_sources(PRIVATE ${headers} ${sources})`**。  
- **包含目录**：工程根、**`modules/core|imgproc|features2d/include`**。  
- **无** 额外 **`target_link_libraries`**（纯编译单元 + 标准/内置向量库依赖由工具链提供）。  
- **导出**：**`RVV_HAL_VERSION`** = **`0.0.1`**，**`RVV_HAL_LIBRARIES`** = **`rvv_hal`**，**`RVV_HAL_HEADERS`** = **`rvv_hal.hpp`**，**`RVV_HAL_INCLUDE_DIRS`** = **本目录**。

---

## 4. 头文件中的 HAL 组织方式

- **命名空间**：**`cv::rvv_hal::core`**、**`cv::rvv_hal::imgproc`**、**`cv::rvv_hal::features2d`**。  
- **对接方式**：在 **`core.hpp` / `imgproc.hpp`** 内 **`#undef cv_hal_xxx`** / **`#define cv_hal_xxx cv::rvv_hal::…`**，与 IPP、NDSRVP 等模式一致。  
- **`include/types.hpp`**：在 **`CV_HAL_RVV_1P0_ENABLED`** 下定义 **`RVV_LMUL`**、**`RVV<T,LMUL>`** 别名及 **`RVV_SameLen`** 等辅助模板，供 **`src`** 中向量化抽象复用。  
- **实现侧公共内建**：如 **`src/core/common.hpp`** 中 **`CV_HAL_RVV_COMMON_*`** 宏封装 **`__riscv_vabd`**、自定义 **`__riscv_vabs`**、**`__riscv_vfrec`** 等。

---

## 5. 刻意关闭或未接线的 HAL（阅读源码时注意）

以下在头文件中有说明或注释，**不一定**走 RVV：

- **`imgproc.hpp`**：**`imgproc::integral`** 已实现，但 **`cv_hal_integral`** 映射到 **`cv::rvv_hal::imgproc::integral`** 被整体注释，注明 **accuracy issue** 并指向 **GitHub #27407**；因此 **imgproc 模块的积分图 HAL** 当前不通过本头文件走 RVV。  
- **`imgproc.hpp`**：普通 **`cv_hal_threshold`** 的 **`#define`** 被注释，说明仅 **UI 路径足够快**；保留 **`cv_hal_threshold_otsu`**、**`cv_hal_adaptiveThreshold`**。  
- **`core.hpp`**：**`cv_hal_div64f` / `recip64f` / `cmp64f`** 等部分 **64 位** 入口以 **`//` 注释**掉宏。  

若以 **`grep "^#undef cv_hal"`** 核对，可与 **OpenCV 默认实现** 回退路径对照。

---

## 6. `features2d.hpp` 与 FAST

- **`#if CV_HAL_RVV_1P0_ENABLED`**：**`FAST`** 对应 **`cv_hal_FASTv2`**（参数含 **`realloc_func`**，与新版 HAL 接口一致）。  
- 文件末尾 **guard 注释**写成了 **`OPENCV_RVV_HAL_IMGPROC_HPP`**，与 **`#ifndef OPENCV_RVV_HAL_FEATURES2D_HPP`** 不一致，属笔误级问题，不影响 **`#pragma once`** 语义缺失时仍以 **`#ifndef`** 为准。

---

## 7. 与根 **`OpenCV_HAL` 顺序**

**`rvvhal`** 与 **NDSRVP、KleidiCV、IPP** 等一样可在配置中 **prepend**；最终是否命中 RVV 实现仍取决于 **`cv_hal_*` 返回值** 及各模块调用顺序。

---

## 8. 推荐阅读顺序

1. 根 **`CMakeLists.txt`**：**`WITH_HAL_RVV`**、**`rvvhal` 分支**与 **`CPU_BASELINE_FINAL`** 中的 **RVV**。  
2. **`rvv_hal.hpp`**：**`CV_HAL_RVV_1P0_ENABLED` / `CV_HAL_RVV_071_ENABLED`**。  
3. **`include/types.hpp`**（RVV 1.0 段）：类型与 **LMUL** 约定。  
4. **`include/core.hpp`、`include/imgproc.hpp`**：完整 **HAL 替换表** 与注释掉的入口。  
5. **按需阅读 **`src/core/*.cpp`**、**`src/imgproc/*.cpp`** 与 **`src/core/common.hpp`**。

---

## 9. 版权与来源（头文件标注）

**`types.hpp`** 等标注 **Institute of Software, Chinese Academy of Sciences**；**`src/core/common.hpp`** 另含 **SpaceMIT Inc.** 等 **Copyright** 行。以各文件头为准。

---

## 10. 路径与版本说明

- 分析对象：`/home/work2/ImgAlgo/opencv-4.13.0/hal/riscv-rvv`。  
- **GCC/Clang 的 `__riscv_v` 数值**、**VLEN**、**LMUL** 与 **`-march=…_zve64d`** 等标志须与目标芯片一致；否则易出现 **编译失败** 或 **未进入 1P0 分支**。

---

*文档用于源码导航；RVV HAL 与 OpenCV 版本演进可能增减算子或调整宏开关，以当前树为准。*
