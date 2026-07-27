/**
 * 06_edge —— 边缘检测
 * 目标：Sobel / Laplacian / Canny
 */
#include "../common/demo_utils.hpp"
#include <iostream>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0], "[--canny_low=50] [--canny_high=150]");
    cv::Mat src = demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "06_edge");

    cv::Mat gray, blur;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 1.2);

    cv::Mat gx, gy, sobel;
    cv::Sobel(blur, gx, CV_16S, 1, 0, 3);
    cv::Sobel(blur, gy, CV_16S, 0, 1, 3);
    cv::Mat absx, absy;
    cv::convertScaleAbs(gx, absx);
    cv::convertScaleAbs(gy, absy);
    cv::addWeighted(absx, 0.5, absy, 0.5, 0, sobel);

    cv::Mat lap, lapAbs;
    cv::Laplacian(blur, lap, CV_16S, 3);
    cv::convertScaleAbs(lap, lapAbs);

    int t1 = std::stoi(demo::getArg(argc, argv, "--canny_low", "50"));
    int t2 = std::stoi(demo::getArg(argc, argv, "--canny_high", "150"));
    cv::Mat canny;
    cv::Canny(blur, canny, t1, t2);

    // 把 Canny 画回彩色图
    cv::Mat overlay = src.clone();
    overlay.setTo(cv::Scalar(0, 255, 255), canny);

    demo::save(dir, "06_sobel.png", sobel);
    demo::save(dir, "06_laplacian.png", lapAbs);
    demo::save(dir, "06_canny.png", canny);
    demo::save(dir, "06_canny_overlay.png", overlay);

    demo::showIfRequested(argc, argv, "Canny", canny);
    demo::showIfRequested(argc, argv, "Overlay", overlay);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 07_morphology\n";
    return 0;
}
