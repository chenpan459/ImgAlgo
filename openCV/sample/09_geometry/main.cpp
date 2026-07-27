/**
 * 09_geometry —— 几何变换
 * 目标：resize / flip / getRotationMatrix2D / warpAffine / getPerspectiveTransform / warpPerspective
 */
#include "../common/demo_utils.hpp"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    cv::Mat src = demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "09_geometry");

    cv::Mat small, flipped, rotated;
    cv::resize(src, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::flip(src, flipped, 1);  // 水平翻转

    cv::Point2f center(src.cols / 2.f, src.rows / 2.f);
    cv::Mat M = cv::getRotationMatrix2D(center, 30.0, 0.9);
    cv::warpAffine(src, rotated, M, src.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // 透视：把图像中心区域“拉正/拉斜”
    std::vector<cv::Point2f> srcPts = {
        {100.f, 80.f}, {src.cols - 80.f, 60.f}, {src.cols - 60.f, src.rows - 70.f}, {90.f, src.rows - 50.f}};
    std::vector<cv::Point2f> dstPts = {
        {0.f, 0.f}, {400.f, 0.f}, {400.f, 300.f}, {0.f, 300.f}};
    cv::Mat H = cv::getPerspectiveTransform(srcPts, dstPts);
    cv::Mat warped;
    cv::warpPerspective(src, warped, H, cv::Size(400, 300));

    cv::Mat marked = src.clone();
    for (int i = 0; i < 4; ++i) {
        cv::line(marked, srcPts[i], srcPts[(i + 1) % 4], cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        cv::circle(marked, srcPts[i], 4, cv::Scalar(0, 0, 255), -1);
    }

    demo::save(dir, "09_half.png", small);
    demo::save(dir, "09_flip.png", flipped);
    demo::save(dir, "09_rotate30.png", rotated);
    demo::save(dir, "09_perspective_src.png", marked);
    demo::save(dir, "09_perspective_dst.png", warped);

    demo::showIfRequested(argc, argv, "Rotate", rotated);
    demo::showIfRequested(argc, argv, "Warp", warped);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 10_feature_match\n";
    return 0;
}
