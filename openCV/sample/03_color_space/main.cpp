/**
 * 03_color_space —— 颜色空间
 * 目标：BGR / Gray / HSV / LAB，以及 HSV 颜色阈值
 */
#include "../common/demo_utils.hpp"
#include <iostream>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    cv::Mat bgr = demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "03_color_space");

    cv::Mat gray, hsv, lab;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);

    // 提取偏红色区域（OpenCV H: 0-180）
    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, cv::Scalar(0, 80, 60), cv::Scalar(10, 255, 255), mask1);
    cv::inRange(hsv, cv::Scalar(170, 80, 60), cv::Scalar(180, 255, 255), mask2);
    mask = mask1 | mask2;

    cv::Mat redOnly;
    bgr.copyTo(redOnly, mask);

    // 可视化 HSV 的 H 通道
    std::vector<cv::Mat> hsvCh;
    cv::split(hsv, hsvCh);
    cv::Mat hVis;
    hsvCh[0].convertTo(hVis, CV_8U, 255.0 / 180.0);

    demo::save(dir, "03_bgr.png", bgr);
    demo::save(dir, "03_gray.png", gray);
    demo::save(dir, "03_h_channel.png", hVis);
    demo::save(dir, "03_red_mask.png", mask);
    demo::save(dir, "03_red_only.png", redOnly);

    demo::showIfRequested(argc, argv, "HSV-H", hVis);
    demo::showIfRequested(argc, argv, "Red", redOnly);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 04_drawing\n";
    return 0;
}
