/**
 * 02_mat_roi —— Mat 创建、像素访问、ROI
 * 目标：理解连续内存、at<>、clone vs 引用 ROI
 */
#include "../common/demo_utils.hpp"
#include <iostream>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    std::string dir = demo::outDir(argc, argv, "02_mat_roi");

    // 1) 创建纯色 Mat
    cv::Mat canvas(240, 320, CV_8UC3, cv::Scalar(30, 30, 30));

    // 2) 像素级写入：画渐变条
    for (int y = 0; y < canvas.rows; ++y) {
        for (int x = 0; x < canvas.cols; ++x) {
            canvas.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>(x * 255 / canvas.cols),
                static_cast<uchar>(y * 255 / canvas.rows),
                128);
        }
    }

    // 3) ROI 是引用：修改 roi 会改原图
    cv::Rect box(40, 40, 100, 80);
    cv::Mat roi = canvas(box);
    roi.setTo(cv::Scalar(0, 0, 255));  // 红块

    // 4) clone 才是独立拷贝
    cv::Mat roiCopy = canvas(cv::Rect(180, 40, 100, 80)).clone();
    roiCopy.setTo(cv::Scalar(0, 255, 0));  // 不会影响原图
    roiCopy.copyTo(canvas(cv::Rect(180, 140, 100, 80)));

    // 5) 指针行访问（更快）
    for (int y = 200; y < 220; ++y) {
        auto* row = canvas.ptr<cv::Vec3b>(y);
        for (int x = 0; x < canvas.cols; ++x) row[x] = cv::Vec3b(255, 255, 255);
    }

    demo::save(dir, "02_mat_roi.png", canvas);
    demo::showIfRequested(argc, argv, "Mat ROI", canvas);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 03_color_space\n";
    return 0;
}
