/**
 * 18_watershed —— 分水岭分割（高级）
 * 目标：距离变换 + 必经标记 + watershed
 */
#include "../common/demo_utils.hpp"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    cv::Mat src = demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "18_watershed");

    cv::Mat gray, bin, sureBg, dist, sureFg, unknown, markers;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(bin, sureBg, kernel, cv::Point(-1, -1), 3);

    cv::distanceTransform(bin, dist, cv::DIST_L2, 5);
    double maxVal;
    cv::minMaxLoc(dist, nullptr, &maxVal);
    cv::threshold(dist, sureFg, 0.4 * maxVal, 255, cv::THRESH_BINARY);
    sureFg.convertTo(sureFg, CV_8U);

    cv::subtract(sureBg, sureFg, unknown);
    cv::connectedComponents(sureFg, markers);
    markers += 1;  // 背景记为 1，未知为 0
    markers.setTo(0, unknown);

    cv::Mat vis = src.clone();
    cv::watershed(vis, markers);

    // 边界为 -1，着色各区域
    cv::Mat overlay = src.clone();
    overlay.setTo(cv::Scalar(0, 0, 255), markers == -1);
    cv::Mat colorLabels(src.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::RNG rng(12345);
    double minM, maxM;
    cv::minMaxLoc(markers, &minM, &maxM);
    for (int i = 2; i <= static_cast<int>(maxM); ++i) {
        cv::Vec3b c(rng.uniform(40, 255), rng.uniform(40, 255), rng.uniform(40, 255));
        colorLabels.setTo(c, markers == i);
    }

    cv::Mat distVis;
    cv::normalize(dist, distVis, 0, 255, cv::NORM_MINMAX);
    distVis.convertTo(distVis, CV_8U);

    demo::save(dir, "18_bin.png", bin);
    demo::save(dir, "18_dist.png", distVis);
    demo::save(dir, "18_boundary.png", overlay);
    demo::save(dir, "18_regions.png", colorLabels);

    demo::showIfRequested(argc, argv, "Watershed", overlay);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 19_grabcut\n";
    return 0;
}
