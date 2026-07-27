/**
 * 01_hello_image —— OpenCV 入门第一课
 * 目标：imread / imwrite / Mat 基本信息 / 灰度转换
 */
#include "../common/demo_utils.hpp"
#include <iostream>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    cv::Mat bgr = demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "01_hello_image");

    std::cout << "type=" << bgr.type() << " channels=" << bgr.channels()
              << " depth=" << bgr.depth() << " empty=" << bgr.empty() << std::endl;

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    demo::save(dir, "01_bgr.png", bgr);
    demo::save(dir, "01_gray.png", gray);
    demo::showIfRequested(argc, argv, "BGR", bgr);
    demo::showIfRequested(argc, argv, "Gray", gray);
    demo::waitIfShown(argc, argv);

    std::cout << "Done. Next: 02_mat_roi\n";
    return 0;
}
