/**
 * 05_filter —— 图像滤波
 * 目标：blur / GaussianBlur / medianBlur / bilateralFilter
 */
#include "../common/demo_utils.hpp"
#include <iostream>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    cv::Mat src = demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "05_filter");

    cv::Mat meanB, gauss, median, bilateral;
    cv::blur(src, meanB, cv::Size(7, 7));
    cv::GaussianBlur(src, gauss, cv::Size(7, 7), 1.5);
    cv::medianBlur(src, median, 5);
    cv::bilateralFilter(src, bilateral, 9, 75, 75);

    // 拼成 2x2 对照
    cv::Mat top, bottom, grid, r1, r2, r3, r4;
    cv::resize(src, r1, cv::Size(320, 240));
    cv::resize(gauss, r2, cv::Size(320, 240));
    cv::resize(median, r3, cv::Size(320, 240));
    cv::resize(bilateral, r4, cv::Size(320, 240));
    cv::putText(r1, "src", {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 255}, 2);
    cv::putText(r2, "gaussian", {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 255}, 2);
    cv::putText(r3, "median", {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 255}, 2);
    cv::putText(r4, "bilateral", {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 255}, 2);
    cv::hconcat(r1, r2, top);
    cv::hconcat(r3, r4, bottom);
    cv::vconcat(top, bottom, grid);

    demo::save(dir, "05_src.png", src);
    demo::save(dir, "05_gaussian.png", gauss);
    demo::save(dir, "05_median.png", median);
    demo::save(dir, "05_bilateral.png", bilateral);
    demo::save(dir, "05_compare.png", grid);

    demo::showIfRequested(argc, argv, "Filter Compare", grid);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 06_edge\n";
    return 0;
}
