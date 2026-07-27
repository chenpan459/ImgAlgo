/**
 * 19_grabcut —— GrabCut 前景分割（高级）
 * 目标：用矩形初始化 grabCut，提取前景 mask
 */
#include "../common/demo_utils.hpp"
#include <iostream>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    cv::Mat src = demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "19_grabcut");

    // 默认框住图像中心偏上的主体；可用 --rect=x,y,w,h 覆盖
    cv::Rect rect(80, 60, src.cols - 160, src.rows - 160);
    std::string rectStr = demo::getArg(argc, argv, "--rect");
    if (!rectStr.empty()) {
        int x, y, w, h;
        if (sscanf(rectStr.c_str(), "%d,%d,%d,%d", &x, &y, &w, &h) == 4) {
            rect = cv::Rect(x, y, w, h) & cv::Rect(0, 0, src.cols, src.rows);
        }
    }

    cv::Mat mask(src.size(), CV_8UC1, cv::GC_BGD);
    mask(rect).setTo(cv::GC_PR_FGD);
    cv::Mat bgModel, fgModel;
    cv::grabCut(src, mask, rect, bgModel, fgModel, 5, cv::GC_INIT_WITH_RECT);

    cv::Mat fgMask = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);
    cv::Mat foreground(src.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    src.copyTo(foreground, fgMask);

    cv::Mat rectVis = src.clone();
    cv::rectangle(rectVis, rect, {0, 255, 255}, 2);

    demo::save(dir, "19_rect.png", rectVis);
    demo::save(dir, "19_mask.png", fgMask * 255);
    demo::save(dir, "19_foreground.png", foreground);

    demo::showIfRequested(argc, argv, "GrabCut FG", foreground);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 20_stitching\n";
    return 0;
}
