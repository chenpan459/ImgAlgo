# OpenCV 4.13.0 — `samples` 目录与构建说明

本文档概括 **`opencv-4.13.0/samples`**：官方示例与教程配套源码的集合，涵盖 **C++、Python、Java、Android、DNN、GPU** 等；多数通过 **`samples/CMakeLists.txt`** 在配置 **`BUILD_EXAMPLES`（默认 OFF）** 时编入工程。另有 **Standalone 模式**（将 **`samples` 作为顶层** **`cmake`** **指向已安装的 OpenCV**）。

---

## 1. 顶层 `CMakeLists.txt` 双模式

| 模式 | 条件 | 行为摘要 |
|------|------|----------|
| **内嵌构建** | **`CMAKE_SOURCE_DIR` ≠ 当前目录**（随 OpenCV 主工程配置） | **`add_subdirectory(cpp|java/tutorial_code|dnn|gpu|…)`** 等；**`include(samples_utils.cmake)`**；可选安装 **`data`**、**`INSTALL_C_EXAMPLES`** 下安装源码。 |
| **独立构建** | **`else()` 分支**：本文件作 **project root** | **`find_package(OpenCV REQUIRED)`**；默认 **`add_subdirectory(cpp dnn opencl sycl tapi)`** 等；**注释掉** **gpu / opengl / openvx / va_intel**（需自行按需打开）。 |

根 **`CMakeLists.txt`** 中与示例相关选项（节选）：

- **`BUILD_EXAMPLES`**：是否构建示例（默认 **OFF**）。
- **`INSTALL_C_EXAMPLES` / `INSTALL_PYTHON_EXAMPLES`**：是否把 **C / Python** 示例**源码**安装到 **`OPENCV_SAMPLES_SRC_INSTALL_PATH`**。
- **`INSTALL_BIN_EXAMPLES`**：**Windows** 下与预编译示例安装相关（**`WIN32 IF BUILD_EXAMPLES`**）。

**线程**：若存在 **`Threads::Threads`**（或 **MSVC/APPLE** 等），定义 **`HAVE_THREADS=1`**，供示例源使用。

---

## 2. 子目录与 inclusion 条件（内嵌构建）

根级 **`samples/CMakeLists.txt`** 中 **`add_subdirectory` 与条件**如下（与 OpenCV 功能开关绑定）：

| 子目录 | 条件或说明 |
|--------|------------|
| **`cpp`** | 始终加入（依赖满足时实际生成目标）。 |
| **`java/tutorial_code`** | 始终加入。 |
| **`dnn`** | 始终加入。 |
| **`gpu`** | 始终加入（**CUDA 模块**不足时 **`cpp` 内会过滤 `*/gpu/`** 源文件）。 |
| **`tapi`** | 始终加入。 |
| **`opencl`** | 始终加入。 |
| **`sycl`** | 始终加入。 |
| **`directx`** | **`WIN32` 且 `HAVE_DIRECTX`**。 |
| **`opengl`** | **非 ANDROID 且 `HAVE_OPENGL`**。 |
| **`openvx`** | **`HAVE_OPENVX`**。 |
| **`va_intel`** | **UNIX、非 ANDROID 且 `HAVE_VA`**（Intel VA 相关）。 |
| **`android`** | **ANDROID 且（`BUILD_ANDROID_EXAMPLES` 或 `INSTALL_ANDROID_EXAMPLES`）**。 |
| **`python`** | **`INSTALL_PYTHON_EXAMPLES`**（主要做 **`.py` 安装**）。 |
| **`semihosting`** | **`OPENCV_SEMIHOSTING`**（ARM 半主机调试场景，见 CMake 内 Arm 文档链接）。 |

**说明**：**`samples/hal`**（自定义 HAL 演示）、**`wp8` / `winrt` / `swift` / `gdb`** 等**不**出现在上述根列表中，多为**历史平台**或**独立文档/工程**，需单独打开或按各自 **README** 使用。

---

## 3. C++ 示例（`samples/cpp`）

- **发现方式**：**`file(GLOB_RECURSE … *.cpp)`**，**每个 `.cpp` 对应一个可执行文件**。  
- **目标命名**（**`samples_utils.cmake`** **`ocv_define_sample`**）：**`example_<group>_<文件名无后缀>`**，其中 **group** 为 **`cpp`**、**`tutorial`** 或 **`snippet`**（路径含 **`tutorial_code`** / **`snippet`** 时）。  
- **聚合目标**：**`opencv_samples_cpp`**、**`opencv_samples_tutorial`** 等，并挂到 **`opencv_samples`**。  
- **默认链接模块**：**core、imgproc、videoio、highgui、dnn、gapi** 等一长串（见 **`OPENCV_CPP_SAMPLES_REQUIRED_DEPS`**）；单样本可用 **`DEPS_<target>`** 覆盖依赖。  
- **过滤**：无 CUDA 相关模块时排除 **`/gpu/`** 下源；排除 **`real_time_pose_estimation/`**、**`parallel_backend/`**（后者在 **CMake≥3.9** 且未设 **`OPENCV_EXAMPLES_SKIP_PARALLEL_BACKEND`** 时 **单独 `add_subdirectory`**）。  
- **特例**：**`tutorial_code/calib3d/real_time_pose_estimation`** 通过 **`include(… OPTIONAL)`**；**`simd_*`** 示例默认不启用自定义 SIMD 头（注释说明为演示保留）。  
- **规模参考**：树内约 **230+** 个 **`cpp` 文件**（含 **`tutorial_code`** 与 **snippets**）。  

**`cpp/example_cmake`**：展示 **`find_package(OpenCV)`** 的**最小工程**（**Standalone + CMake≥3.1** 时加入）。

**`samples/data`**：公共测试图片/数据；**`INSTALL_C_EXAMPLES`** 时随 **`samples_data`** 组件安装。

---

## 4. DNN 示例（`samples/dnn`）

- 独立 **`CMakeLists.txt`**：**`GLOB_RECURSE *.cpp`**，依赖 **dnn、objdetect、video** 等。  
- 目标前缀 **`example_dnn_*`**。  
- 子目录含 **face_detector、dnn_model_runner** 及 **`results`** 等脚本/资源，以仓库实际内容为准。

---

## 5. GPU / OpenCL / SYCL / TAPI

- **`gpu`**：依赖 **CUDA** 系列模块（**`opencv_cudaarithm`** 等），**`HAVE_CUDA`** 时定义宏。  
- **`opencl`、`sycl`、`tapi`**：各自 **`CMakeLists.txt`** 定义可执行文件与依赖；用于 **OpenCL UMat、SYCL、透明 API** 等演示。

---

## 6. Python（`samples/python`）

- 根 **内嵌模式**下：仅当 **`INSTALL_PYTHON_EXAMPLES`** 将 **`*.py`** 安装到 **`${OPENCV_SAMPLES_SRC_INSTALL_PATH}/python`**。  
- **`tutorial_code`** 等目录与官方 Python 教程对应；数量级约 **百余个** **`.py`** 文件。

---

## 7. Java（`samples/java`）

- **CMake 构建**主要包含 **`java/tutorial_code`**（与 Java 教程配套）。  
- **`java/ant`、`eclipse`、`sbt`、`clojure`** 等为**各工具链/语言的辅助工程**，不都在根 **`samples/CMakeLists.txt`** 里统一 **`add_subdirectory`**，按目录内说明使用。

---

## 8. Android（`samples/android`）

- 在 **Android 且** 打开 **`BUILD_ANDROID_EXAMPLES` 或 `INSTALL_ANDROID_EXAMPLES`** 时 **`add_subdirectory(android)`**。  
- 含 **相机预览、tutorial、人脸检测、Mobilenet、二维码、15-puzzle** 等 **Gradle/NDK** 示例。

---

## 9. 自定义 HAL 示例（`samples/hal`，未挂主 CMake）

- 见 **`samples/hal/README.md`**：**`c_hal`**（纯 **C** 占位实现，用于错误路径验证）、**`slow_hal`**（故意缓慢的位运算实现）。  
- **独立 CMake 工程**：例如 **`cmake …/samples/hal/slow_hal`** 生成静态库，再通过 **`-DOpenCV_HAL_DIR=...`** 与主 OpenCV 集成；**不**通过 **`samples/CMakeLists.txt`** 自动编译。

---

## 10. 其它历史或独立目录（概要）

| 路径 | 说明 |
|------|------|
| **`wp8`** | Windows Phone 8 **XAML / Direct3D** 老示例工程。 |
| **`winrt` / `winrt_universal`** | WinRT / 通用 Windows 示例。 |
| **`swift/ios`** | iOS / Swift 相关示例。 |
| **`install`** | 与安装/打包辅助相关脚本或占位（以目录内文件为准）。 |

这些目录通常**不**随默认 **`BUILD_EXAMPLES`** 从 **`samples/CMakeLists.txt`** 一键构建，适用于档案参考或按平台单独维护。

---

## 11. 阅读与调试建议

1. 需要编示例：配置 **`-DBUILD_EXAMPLES=ON`**，并保证所需模块（**CUDA/OpenGL/OpenVX** 等）已开启。  
2. 查 **`example_cpp_*` / `example_dnn_*`** 目标：CMake 配置阶段可设 **`OPENCV_DUMP_EXAMPLE_TARGET`** 打印目标与源文件对应（见 **`ocv_define_sample`**）。  
3. 教程对应：优先 **`cpp/tutorial_code`**、**`python/tutorial_code`**、**`java/tutorial_code`** 与官方文档章节同步。

---

## 12. 路径与版本

- 分析对象：`/home/work2/ImgAlgo/opencv-4.13.0/samples`。  
- 各发行版可能增减示例文件；**`.cpp` 数量**以当前树 **`find`** 为准。

---

*文档用于快速导航 samples 树与 CMake 行为；具体单个示例语义以源文件头注释与官方教程为准。*
