/**
 * 07_morphology —— 形态学
 * 目标：erode / dilate / open / close / gradient / top-hat
 */
#include "../common/demo_utils.hpp"
#include <iostream>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    cv::Mat src = demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "07_morphology");

    cv::Mat gray, bin;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat eroded, dilated, opened, closed, grad, tophat;
    cv::erode(bin, eroded, kernel);
    cv::dilate(bin, dilated, kernel);
    cv::morphologyEx(bin, opened, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(bin, closed, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(bin, grad, cv::MORPH_GRADIENT, kernel);
    cv::morphologyEx(bin, tophat, cv::MORPH_TOPHAT, kernel);

    demo::save(dir, "07_bin.png", bin);
    demo::save(dir, "07_erode.png", eroded);
    demo::save(dir, "07_dilate.png", dilated);
    demo::save(dir, "07_open.png", opened);
    demo::save(dir, "07_close.png", closed);
    demo::save(dir, "07_gradient.png", grad);
    demo::save(dir, "07_tophat.png", tophat);

    demo::showIfRequested(argc, argv, "Open", opened);
    demo::showIfRequested(argc, argv, "Close", closed);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 08_contour\n";
    return 0;
}
