/**
 * 16_optical_flow —— 光流（中级）
 * 目标：Lucas-Kanade 稀疏光流 + Farneback 稠密光流可视化
 */
#include "../common/demo_utils.hpp"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    cv::Mat frame1 = demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "16_optical_flow");

    // 第二帧：整体平移，模拟运动
    cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 12, 0, 1, -8);
    cv::Mat frame2;
    cv::warpAffine(frame1, frame2, M, frame1.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);

    cv::Mat g1, g2;
    cv::cvtColor(frame1, g1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame2, g2, cv::COLOR_BGR2GRAY);

    // --- 稀疏 LK ---
    std::vector<cv::Point2f> pts1, pts2;
    cv::goodFeaturesToTrack(g1, pts1, 120, 0.01, 8);
    std::vector<uchar> status;
    std::vector<float> err;
    if (!pts1.empty()) {
        cv::calcOpticalFlowPyrLK(g1, g2, pts1, pts2, status, err);
    }
    cv::Mat lkVis = frame2.clone();
    int ok = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (!status[i]) continue;
        ++ok;
        cv::arrowedLine(lkVis, pts1[i], pts2[i], {0, 255, 255}, 1, cv::LINE_AA, 0, 0.3);
        cv::circle(lkVis, pts2[i], 2, {0, 0, 255}, -1);
    }

    // --- 稠密 Farneback ---
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(g1, g2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    cv::Mat mag, ang;
    std::vector<cv::Mat> flowXY(2);
    cv::split(flow, flowXY);
    cv::cartToPolar(flowXY[0], flowXY[1], mag, ang, true);
    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
    cv::Mat hsv(frame1.size(), CV_8UC3);
    for (int y = 0; y < hsv.rows; ++y) {
        for (int x = 0; x < hsv.cols; ++x) {
            hsv.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>(ang.at<float>(y, x) * 0.5f),  // H 0-180
                255,
                static_cast<uchar>(mag.at<float>(y, x)));
        }
    }
    cv::Mat flowBgr;
    cv::cvtColor(hsv, flowBgr, cv::COLOR_HSV2BGR);

    std::cout << "LK points tracked=" << ok << "/" << pts1.size() << std::endl;
    demo::save(dir, "16_frame1.png", frame1);
    demo::save(dir, "16_frame2.png", frame2);
    demo::save(dir, "16_lk.png", lkVis);
    demo::save(dir, "16_farneback.png", flowBgr);

    demo::showIfRequested(argc, argv, "LK", lkVis);
    demo::showIfRequested(argc, argv, "Dense", flowBgr);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 17_bg_subtract\n";
    return 0;
}
