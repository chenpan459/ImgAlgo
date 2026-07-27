/**
 * 26_inpaint_clone —— 修复与无缝融合（高级）
 * 目标：inpaint 去污 + seamlessClone 贴图融合
 */
#include "../common/demo_utils.hpp"
#include <opencv2/photo.hpp>
#include <iostream>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    cv::Mat src = demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "26_inpaint_clone");

    // --- inpaint：人造划痕 ---
    cv::Mat damaged = src.clone();
    cv::line(damaged, {50, 80}, {580, 200}, {0, 0, 0}, 6, cv::LINE_AA);
    cv::circle(damaged, {320, 300}, 18, {0, 0, 0}, -1);
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::line(mask, {50, 80}, {580, 200}, 255, 8, cv::LINE_AA);
    cv::circle(mask, {320, 300}, 20, 255, -1);

    cv::Mat inpainted;
    cv::inpaint(damaged, mask, inpainted, 3, cv::INPAINT_TELEA);

    // --- seamlessClone：把绿色圆贴到修复图上 ---
    cv::Mat obj(120, 120, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(obj, {60, 60}, 45, {0, 220, 0}, -1, cv::LINE_AA);
    cv::putText(obj, "OK", {35, 70}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 255, 255}, 2);
    cv::Mat objMask;
    cv::cvtColor(obj, objMask, cv::COLOR_BGR2GRAY);
    cv::threshold(objMask, objMask, 10, 255, cv::THRESH_BINARY);

    cv::Mat cloned;
    cv::Point center(150, 360);
    cv::seamlessClone(obj, inpainted, objMask, center, cloned, cv::MIXED_CLONE);

    // 对照：普通 copyTo（有硬边）
    cv::Mat hard = inpainted.clone();
    cv::Rect roi(center.x - 60, center.y - 60, 120, 120);
    roi &= cv::Rect(0, 0, hard.cols, hard.rows);
    if (roi.width == 120 && roi.height == 120) obj.copyTo(hard(roi), objMask);

    demo::save(dir, "26_damaged.png", damaged);
    demo::save(dir, "26_mask.png", mask);
    demo::save(dir, "26_inpaint.png", inpainted);
    demo::save(dir, "26_clone_seamless.png", cloned);
    demo::save(dir, "26_clone_hard.png", hard);

    demo::showIfRequested(argc, argv, "Inpaint", inpainted);
    demo::showIfRequested(argc, argv, "Seamless", cloned);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Advanced also: DiffImg (image similarity)\n";
    return 0;
}
