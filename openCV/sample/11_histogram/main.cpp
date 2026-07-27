/**
 * 11_histogram —— 直方图与对比度增强（中级）
 * 目标：calcHist / equalizeHist / CLAHE / 直方图可视化
 */
#include "../common/demo_utils.hpp"
#include <iostream>
#include <vector>

static cv::Mat drawHist(const cv::Mat& gray, int histSize = 256) {
    float range[] = {0, 256};
    const float* ranges = {range};
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &ranges);
    cv::normalize(hist, hist, 0, 180, cv::NORM_MINMAX);

    cv::Mat canvas(200, histSize, CV_8UC3, cv::Scalar(20, 20, 20));
    for (int i = 1; i < histSize; ++i) {
        cv::line(canvas,
                 {i - 1, 199 - cvRound(hist.at<float>(i - 1))},
                 {i, 199 - cvRound(hist.at<float>(i))},
                 {0, 255, 255}, 1, cv::LINE_AA);
    }
    return canvas;
}

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    cv::Mat src = demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "11_histogram");

    cv::Mat gray, eq, claheOut;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, eq);

    auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray, claheOut);

    // 对彩色图只均衡化亮度（YUV）
    cv::Mat yuv, yuvEq;
    cv::cvtColor(src, yuv, cv::COLOR_BGR2YUV);
    std::vector<cv::Mat> ch;
    cv::split(yuv, ch);
    cv::equalizeHist(ch[0], ch[0]);
    cv::merge(ch, yuvEq);
    cv::Mat colorEq;
    cv::cvtColor(yuvEq, colorEq, cv::COLOR_YUV2BGR);

    demo::save(dir, "11_gray.png", gray);
    demo::save(dir, "11_equalize.png", eq);
    demo::save(dir, "11_clahe.png", claheOut);
    demo::save(dir, "11_color_eq.png", colorEq);
    demo::save(dir, "11_hist_before.png", drawHist(gray));
    demo::save(dir, "11_hist_after.png", drawHist(eq));

    demo::showIfRequested(argc, argv, "CLAHE", claheOut);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 12_hough\n";
    return 0;
}
