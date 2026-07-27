#!/usr/bin/env bash
# OpenCV sample 一键编译脚本
# 用法:
#   ./build.sh              # 配置并编译全部 demo
#   ./build.sh clean        # 清理后重新编译
#   ./build.sh run          # 编译并运行 01–10
#   ./build.sh -j8          # 指定并行数
#   ./build.sh --opencv-dir /path/to/lib/cmake/opencv4
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT}/build"
JOBS="$(nproc 2>/dev/null || echo 4)"
OPENCV_DIR="${OpenCV_DIR:-}"
DO_CLEAN=0
DO_RUN=0
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] [clean|run]

Options:
  -j N                 parallel jobs (default: nproc)
  --opencv-dir DIR     OpenCVConfig.cmake 所在目录 (或设环境变量 OpenCV_DIR)
  --debug              CMAKE_BUILD_TYPE=Debug
  -h, --help           show help

Commands:
  (none)               configure + build
  clean                remove build/ then configure + build
  run                  build then run 01-10 demos (write to sample/output/)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -j)
            JOBS="$2"
            shift 2
            ;;
        -j*)
            JOBS="${1#-j}"
            shift
            ;;
        --opencv-dir)
            OPENCV_DIR="$2"
            shift 2
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        clean)
            DO_CLEAN=1
            shift
            ;;
        run)
            DO_RUN=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "${DO_CLEAN}" -eq 1 ]]; then
    echo "[*] Cleaning ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE="${BUILD_TYPE}")
if [[ -n "${OPENCV_DIR}" ]]; then
    CMAKE_ARGS+=(-DOpenCV_DIR="${OPENCV_DIR}")
    echo "[*] Using OpenCV_DIR=${OPENCV_DIR}"
fi

echo "[*] Configuring (build type: ${BUILD_TYPE})"
cmake "${CMAKE_ARGS[@]}" "${ROOT}"

echo "[*] Building with -j${JOBS}"
cmake --build . -j"${JOBS}"

echo
echo "[*] Done. Binaries: ${BUILD_DIR}/bin/"
ls -1 "${BUILD_DIR}/bin" 2>/dev/null || true
echo
echo "Examples:"
echo "  ${BUILD_DIR}/bin/01_hello_image --outdir ${ROOT}/output"
echo "  ${BUILD_DIR}/bin/06_edge --image /path/to.jpg --show"
echo "  cmake --build ${BUILD_DIR} --target run_all_demos"

if [[ "${DO_RUN}" -eq 1 ]]; then
    echo
    echo "[*] Running all beginner demos..."
    cmake --build . --target run_all_demos
    echo "[*] Outputs: ${ROOT}/output/"
fi
