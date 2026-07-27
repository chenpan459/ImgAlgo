/**
 * 15_connected_components —— 连通域分析（中级）
 * 目标：connectedComponentsWithStats / 按面积过滤 / 着色标签图
 */
#include "../common/demo_utils.hpp"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0], "[--min_area=300]");
    cv::Mat src = demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "15_connected_components");
    int minArea = std::stoi(demo::getArg(argc, argv, "--min_area", "300"));

    cv::Mat gray, bin;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.0);
    cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);

    cv::Mat labelColor(src.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat vis = src.clone();
    cv::RNG rng(0xFFFFFFFF);

    int kept = 0;
    for (int i = 1; i < n; ++i) {  // 0 = 背景
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < minArea) continue;
        ++kept;

        cv::Vec3b color(rng.uniform(50, 255), rng.uniform(50, 255), rng.uniform(50, 255));
        labelColor.setTo(color, labels == i);

        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        cv::rectangle(vis, cv::Rect(x, y, w, h), color, 2);
        cv::Point c(cvRound(centroids.at<double>(i, 0)), cvRound(centroids.at<double>(i, 1)));
        cv::circle(vis, c, 3, color, -1);
        cv::putText(vis, "A=" + std::to_string(area), {x, std::max(15, y - 4)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1);
    }

    std::cout << "components=" << (n - 1) << " kept(area>=" << minArea << ")=" << kept << std::endl;
    demo::save(dir, "15_bin.png", bin);
    demo::save(dir, "15_labels.png", labelColor);
    demo::save(dir, "15_stats.png", vis);

    demo::showIfRequested(argc, argv, "CC", vis);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 16_optical_flow\n";
    return 0;
}
