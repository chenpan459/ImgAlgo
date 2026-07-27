/**
 * 17_bg_subtract —— 背景建模与运动前景（中级）
 * 目标：BackgroundSubtractorMOG2 / 形态学净化 / 前景框
 * 说明：用合成帧序列模拟视频，无需摄像头
 */
#include "../common/demo_utils.hpp"
#include <cmath>
#include <iostream>
#include <vector>

static cv::Mat makeScene(int frameIdx, int w = 640, int h = 480) {
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(45, 45, 45));
    // 静态背景
    cv::rectangle(img, {20, 20, 120, 80}, {80, 80, 160}, -1);
    cv::circle(img, {520, 100}, 40, {60, 140, 60}, -1);
    // 移动物体
    int x = 40 + frameIdx * 18;
    int y = 220 + static_cast<int>(20 * std::sin(frameIdx * 0.35));
    cv::rectangle(img, cv::Rect(x, y, 70, 50), {0, 0, 220}, -1);
    cv::Mat noise(h, w, CV_8UC3);
    cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(6));
    return img + noise;
}

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0], "[--frames=25]");
    std::string dir = demo::outDir(argc, argv, "17_bg_subtract");
    int nFrames = std::stoi(demo::getArg(argc, argv, "--frames", "25"));

    auto sub = cv::createBackgroundSubtractorMOG2(200, 16, true);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

    cv::Mat lastFg, lastVis;
    for (int i = 0; i < nFrames; ++i) {
        cv::Mat frame = makeScene(i);
        cv::Mat fg;
        sub->apply(frame, fg);
        cv::morphologyEx(fg, fg, cv::MORPH_OPEN, kernel);

        cv::Mat vis = frame.clone();
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& c : contours) {
            if (cv::contourArea(c) < 200) continue;
            cv::rectangle(vis, cv::boundingRect(c), {0, 255, 255}, 2);
        }

        // 保存若干关键帧
        if (i == 0 || i == nFrames / 2 || i == nFrames - 1) {
            demo::save(dir, cv::format("17_frame_%02d.png", i), frame);
            demo::save(dir, cv::format("17_fg_%02d.png", i), fg);
            demo::save(dir, cv::format("17_vis_%02d.png", i), vis);
        }
        lastFg = fg;
        lastVis = vis;
    }

    demo::showIfRequested(argc, argv, "FG", lastFg);
    demo::showIfRequested(argc, argv, "Vis", lastVis);
    demo::waitIfShown(argc, argv);
    std::cout << "Processed " << nFrames << " synthetic frames. Next: DiffImg (advanced)\n";
    return 0;
}
