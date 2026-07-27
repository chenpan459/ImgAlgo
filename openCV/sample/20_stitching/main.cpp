/**
 * 20_stitching —— 图像拼接（高级）
 * 目标：从一张图裁出重叠视角，用 Stitcher 拼回全景
 */
#include "../common/demo_utils.hpp"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    cv::Mat src = demo::loadOrToy(argc, argv, 800, 480);
    std::string dir = demo::outDir(argc, argv, "20_stitching");

    // 左右两张带重叠的子图，模拟双视角
    int w = src.cols, h = src.rows;
    int cropW = static_cast<int>(w * 0.62);
    cv::Mat left = src(cv::Rect(0, 0, cropW, h)).clone();
    cv::Mat right = src(cv::Rect(w - cropW, 0, cropW, h)).clone();

    // 轻微扰动右图，更接近真实拍摄差异
    cv::Mat Mr = cv::getRotationMatrix2D(cv::Point2f(right.cols / 2.f, right.rows / 2.f), 1.5, 1.0);
    cv::warpAffine(right, right, Mr, right.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);

    std::vector<cv::Mat> imgs = {left, right};
    cv::Mat pano;
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
    cv::Stitcher::Status status = stitcher->stitch(imgs, pano);

    demo::save(dir, "20_left.png", left);
    demo::save(dir, "20_right.png", right);
    if (status == cv::Stitcher::OK && !pano.empty()) {
        demo::save(dir, "20_pano.png", pano);
        std::cout << "Stitch OK, pano size=" << pano.cols << "x" << pano.rows << std::endl;
        demo::showIfRequested(argc, argv, "Pano", pano);
    } else {
        std::cerr << "Stitch failed, status=" << static_cast<int>(status)
                  << " (try richer texture image with --image)\n";
        // 失败时仍给出简单水平拼接对照
        cv::Mat fallback;
        cv::hconcat(left, right, fallback);
        demo::save(dir, "20_fallback_hconcat.png", fallback);
        demo::showIfRequested(argc, argv, "Fallback", fallback);
    }

    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 21_aruco\n";
    return 0;
}
