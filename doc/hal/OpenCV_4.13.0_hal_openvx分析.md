# OpenCV 4.13.0 — `hal/openvx` 代码结构分析

本文档说明 **`opencv-4.13.0/hal/openvx`**：在 **Khronos OpenVX** 运行时之上，用 **即时模式（Immediate）`vxu*`** 与 **OpenCV HAL 宏**对接的 **`openvx_hal`** 静态库；并包含 **`include/ivx.hpp`**——面向 OpenVX 1.x **C API** 的 **C++11 封装**（头文件实现、引用计数、异常）。

---

## 1. 定位与启用条件

| 项目 | 说明 |
|------|------|
| **上游** | 已安装的 OpenVX：**`OPENVX_ROOT`** 下 **`VX/vx.h`** 与库（**`FindOpenVX.cmake`** 默认在 **`lib`/`bin`** 中查找 **`openvx`**、**`vxu`** 等待选名）。 |
| **CMake** | **`OCV_OPTION(WITH_OPENVX "Include OpenVX support" OFF … VERIFY HAVE_OPENVX)`**；**`WITH_OPENVX=ON`** 时 **`include(cmake/FindOpenVX.cmake)`**。 |
| **`HAVE_OPENVX`** | 需 **`OPENVX_INCLUDE_DIR`** 与 **`OPENVX_LIBRARIES`** 均找到；可选 **`try_compile`** 检测引用计数枚举命名，设置 **`IVX_RENAMED_REFS`**。 |
| **HAL 注册** | **`HAVE_OPENVX`** 时将 **`openvx`** **prepend** 到 **`OpenCV_HAL`**；**`add_subdirectory(hal/openvx)`** 后 **`ocv_hal_register(OPENVX_HAL_*)`**。**无** 额外 CPU 架构门控（与 IPP 类似）。 |
| **顶层保护** | **`hal/openvx/CMakeLists.txt`**：若 **`NOT HAVE_OPENVX`** 则 **直接 `return()`**，不构建子目录。 |

**`hal/README.md`** 给出的典型配置：**`-DOPENVX_ROOT=/path/to/prebuilt/openvx -DWITH_OPENVX=YES`**。

---

## 2. 目录结构

```
hal/openvx/
├── CMakeLists.txt              # HAVE_OPENVX 检测失败则 return；add_subdirectory(hal)
├── README.md                   # ivx 封装设计说明（轻量、RAII、异常、C++11）
├── include/
│   ├── ivx.hpp                 # OpenVX C++ 封装（体量大的头文件实现）
│   └── ivx_lib_debug.hpp       # 调试用扩展（如 ivx::debug:: 读写图像等）
└── hal/
    ├── CMakeLists.txt          # 目标 openvx_hal
    ├── README.md               # HAL 构建说明
    ├── openvx_hal.hpp          # ovx_hal_* 声明 + 部分 #undef cv_hal_*
    └── openvx_hal.cpp          # 实现：vxu / Graph 封装、上下文、大量 ovx_hal_*
```

---

## 3. 构建要点（`hal/openvx/hal/CMakeLists.txt`）

- **`add_library(openvx_hal STATIC openvx_hal.cpp openvx_hal.hpp …/ivx.hpp …/ivx_lib_debug.hpp)`**  
- **`target_include_directories` `PUBLIC`**：**本目录 **`hal/`**、**`hal/openvx/include`**、**`modules/core|imgproc|features2d/include`**、**`${OPENVX_INCLUDE_DIR}`**。  
- **`target_link_libraries(openvx_hal PUBLIC ${OPENVX_LIBRARIES})`**。  
- **导出**：**`OPENVX_HAL_VERSION`** = **`0.0.1`**，**`OPENVX_HAL_LIBRARIES`** = **`openvx_hal`**，**`OPENVX_HAL_HEADERS`** = **`openvx_hal.hpp`**，**`OPENVX_HAL_INCLUDE_DIRS`** = **`hal/` + `include/` + OpenVX 头路径**。

---

## 4. `ivx` 封装（`include/ivx.hpp` 等）

根 **`README.md`** 要点：**header-only**、**`ivx` 命名空间**、**自动引用计数**、**`RuntimeError` / `WrapperError`**、OpenVX **1.0 / 1.1** 分支编译；可与 C API 混用。**`ivx_lib_debug.hpp`** 提供示例式调试图读写（需 **`loadKernels("openvx-debug")`** 一类能力，视实现而定）。

---

## 5. HAL 实现要点（`openvx_hal.cpp`）

- **上下文**：**`getOpenVXHALContext()`** 返回 **`thread_local`** **`ivx::Context`**（C++11 / MSVC **`thread_local`** / gcc **`__thread`** 分支），避免多线程共享同一 **vx_context** 的典型问题。  
- **输入/输出缓冲**：常用 **`ivx::Image::createFromHandle`** 将 OpenCV 缓冲区包装为 **vx_image**；派生类 **`vxImage`** 在析构时于 OpenVX **≥1.1** 路径调用 **`swapHandle`**，避免释放句柄与内存不同步。  
- **二元算子宏 `OVX_BINARY_OP`**：对 **add/sub/absdiff/and/or/xor** 调用 **`vxuAdd`**、**`vxuSubtract`** 等；失败时 **`catch`** **`ivx::RuntimeError` / `WrapperError`**，映射为 **`CV_HAL_ERROR_UNKNOWN`**。  
- **小图跳过**：**`skipSmallImages<kernel_id>(w,h)`** 模板特化——注释说明 OpenVX 调用有固定开销，低于某像素数则 **`CV_HAL_ERROR_NOT_IMPLEMENTED`** 回退 OpenCV。  
- **尺寸上限**：**`dimTooBig`** 在厂商为 Khronos/默认时，按 OpenVX **uint32** 寻址与 **`VX_SCALE_UNITY`** 检查宽高。  
- **版本分支**：**`VX_VERSION` / `VX_VERSION_1_0`** 影响边界结构体（如 **`border.constant_value`**）、**morph** 等 **OpenVX 1.1** 特性。  
- **`openvx_hal.hpp` 内注释**：**resize**、**warpAffine**、**warpPerspective**、**sepFilter** 的 **`#define`** 被注释——注明 OpenVX 参考实现与 OpenCV 在 **取整策略**（如 **round to zero** vs **nearest**）上不一致，故不默认替换 **HAL**。  
- **形态学**：**`#if VX_VERSION > VX_VERSION_1_0`** 才在头文件中 **`#undef cv_hal_morph*`**；与 **cpp** 中实现一致。

---

## 6. `openvx_hal.hpp`：宏替换范围（与 cpp 实现对照）

**当前头文件内已启用 `cv_hal_* → ovx_hal_*` 的入口**大致为：

- **Core 类**：**add/sub/absdiff（8u/16s）**、**and/or/xor/not（8u）**、**mul（8u/16s）**、**merge8u**、**filter / filterInit / filterFree**、**颜色转换若干**、**integral**、**meanStdDev**、**lut**、**minMaxIdxMaskStep**。  
- **OpenVX > 1.0**：**morphInit / morph / morphFree**。

**以下在 `openvx_hal.hpp` 中有 `int ovx_hal_*` 声明，且 `openvx_hal.cpp` 中有实现，但同一头文件内未再写 `#undef cv_hal_*`**（即 **未通过本头接入默认 HAL 宏**）：**

**medianBlur、sobel、canny、pyrdown、boxFilter、equalize_hist、gaussianBlur、remap32f、threshold、FAST** 等。

因此：**逻辑实现存在于 `openvx_hal.cpp`**，但若没有其他编译单元在包含 HAL 头之后再次 **`#define`**，OpenCV **imgproc / features2d** 默认 **不会** 通过这些符号走 OpenVX。阅读与集成时请以 **`openvx_hal.hpp` 尾部宏块**为准。

**被注释掉的宏**：**`cv_hal_resize`**、**`cv_hal_warpAffine`**、**`cv_hal_warpPerspective`**、**sepFilter 三件套**（实现仍在 **cpp** 中可能存在，默认不替换）。

---

## 7. 与根 **`CMakeLists.txt` 的衔接**

- **`WITH_OPENVX OR HAVE_OPENVX`** 时在配置摘要中输出 **`OpenVX: YES (${OPENVX_LIBRARIES})`** 等。  
- **`OpenCV_HAL`** 中含 **`openvx`** 时，`hal/openvx` 子目录会被加入并注册。

---

## 8. 推荐阅读顺序

1. **`cmake/FindOpenVX.cmake`** 与 **`OPENVX_ROOT`** 探测逻辑。  
2. **`hal/openvx/README.md`**（**ivx** 用法）→ **`include/ivx.hpp`** 入口 API。  
3. **`hal/openvx/hal/openvx_hal.hpp`**：声明 + **实际启用的 `cv_hal_*` 列表**。  
4. **`hal/openvx/hal/openvx_hal.cpp`**：**`OVX_BINARY_OP`**、**`skipSmallImages`**、各 **`ovx_hal_*`** 与 **vxu** 调用链。

---

## 9. 路径与版本说明

- 分析对象：`/home/work2/ImgAlgo/opencv-4.13.0/hal/openvx`。  
- **OpenVX 1.0 vs 1.1**、厂商实现差异会影响可用核与数值一致性；以目标环境与 **`VX_VERSION`** 为准。

---

*文档用于源码导航；OpenVX 与 OpenCV 版本演进可能调整 HAL 宏与实现覆盖，以当前树为准。*
