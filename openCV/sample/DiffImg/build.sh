#!/usr/bin/env bash
# DiffImg 单独编译脚本
# 用法:
#   ./build.sh
#   ./build.sh clean
#   ./build.sh --opencv-dir /path/to/lib/cmake/opencv4
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT}/build"
JOBS="$(nproc 2>/dev/null || echo 4)"
OPENCV_DIR="${OpenCV_DIR:-}"
DO_CLEAN=0
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] [clean]

Options:
  -j N                 parallel jobs
  --opencv-dir DIR     OpenCV cmake config dir
  --debug              Debug build
  -h, --help           show help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -j) JOBS="$2"; shift 2 ;;
        -j*) JOBS="${1#-j}"; shift ;;
        --opencv-dir) OPENCV_DIR="$2"; shift 2 ;;
        --debug) BUILD_TYPE="Debug"; shift ;;
        clean) DO_CLEAN=1; shift ;;
        *) echo "Unknown: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ "${DO_CLEAN}" -eq 1 ]]; then
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE="${BUILD_TYPE}")
[[ -n "${OPENCV_DIR}" ]] && CMAKE_ARGS+=(-DOpenCV_DIR="${OPENCV_DIR}")

cmake "${CMAKE_ARGS[@]}" "${ROOT}"
cmake --build . -j"${JOBS}"

BIN="${BUILD_DIR}/image_comparison"
# 兼容 RUNTIME_OUTPUT 可能在当前目录
[[ -x "${BIN}" ]] || BIN="$(find "${BUILD_DIR}" -maxdepth 2 -type f -name image_comparison -executable | head -1)"

echo
echo "[*] Done: ${BIN}"
echo "Run: ${BIN} <img1> <img2>"
