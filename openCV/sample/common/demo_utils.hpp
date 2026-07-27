#pragma once
/**
 * 入门 demo 公共工具：命令行、合成测试图、保存/显示结果。
 * 无显示器环境默认只写文件，加 --show 才尝试 imshow。
 */
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace demo {

inline bool hasFlag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == flag) return true;
    }
    return false;
}

inline std::string getArg(int argc, char** argv, const std::string& key, const std::string& def = "") {
    const std::string prefix = key + "=";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind(prefix, 0) == 0) return a.substr(prefix.size());
        if (a == key && i + 1 < argc) return argv[i + 1];
    }
    return def;
}

inline void ensureDir(const std::string& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
}

/** 生成一张带几何图案的 BGR 测试图，便于无外部素材时练习 */
inline cv::Mat makeToyImage(int w = 640, int h = 480) {
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(40, 40, 40));
    cv::rectangle(img, cv::Rect(40, 40, 180, 120), cv::Scalar(0, 0, 220), -1);
    cv::circle(img, cv::Point(420, 160), 70, cv::Scalar(0, 200, 0), -1);
    cv::line(img, cv::Point(80, 360), cv::Point(560, 300), cv::Scalar(220, 180, 0), 4, cv::LINE_AA);
    std::vector<cv::Point> tri = {{300, 380}, {240, 280}, {360, 280}};
    cv::fillConvexPoly(img, tri, cv::Scalar(200, 80, 200));
    cv::putText(img, "OpenCV Demo", cv::Point(200, 60), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    // 轻微噪声，方便看滤波效果
    cv::Mat noise(h, w, CV_8UC3);
    cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(12));
    img = img + noise;
    return img;
}

inline cv::Mat loadOrToy(int argc, char** argv, int w = 640, int h = 480) {
    std::string path = getArg(argc, argv, "--image");
    if (path.empty() && argc > 1 && argv[1][0] != '-') path = argv[1];
    if (!path.empty()) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (!img.empty()) {
            std::cout << "Loaded: " << path << "  " << img.cols << "x" << img.rows << std::endl;
            return img;
        }
        std::cerr << "Failed to load image: " << path << ", fallback to toy image\n";
    }
    std::cout << "Using synthetic toy image " << w << "x" << h << std::endl;
    return makeToyImage(w, h);
}

inline std::string outDir(int argc, char** argv, const std::string& demoName) {
    std::string base = getArg(argc, argv, "--outdir", "output");
    std::string dir = base + "/" + demoName;
    ensureDir(dir);
    return dir;
}

inline void save(const std::string& dir, const std::string& name, const cv::Mat& img) {
    std::string path = dir + "/" + name;
    if (cv::imwrite(path, img)) {
        std::cout << "Saved: " << path << std::endl;
    } else {
        std::cerr << "Failed to save: " << path << std::endl;
    }
}

inline void showIfRequested(int argc, char** argv, const std::string& win, const cv::Mat& img) {
    if (!hasFlag(argc, argv, "--show")) return;
    cv::imshow(win, img);
}

inline void waitIfShown(int argc, char** argv) {
    if (!hasFlag(argc, argv, "--show")) return;
    std::cout << "Press any key in an image window to exit...\n";
    cv::waitKey(0);
}

inline void printHelpHint(const char* prog, const char* extra = "") {
    std::cout << "Usage: " << prog << " [--image path] [--outdir dir] [--show] " << extra << "\n";
}

}  // namespace demo
