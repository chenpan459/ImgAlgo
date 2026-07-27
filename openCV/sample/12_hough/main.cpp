/**
 * 12_hough —— 霍夫变换检测直线与圆（中级）
 * 目标：Canny + HoughLinesP / HoughCircles
 */
#include "../common/demo_utils.hpp"
#include <iostream>
#include <vector>

static cv::Mat makeHoughScene(int w = 640, int h = 480) {
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::line(img, {40, 80}, {600, 120}, {220, 220, 220}, 3, cv::LINE_AA);
    cv::line(img, {60, 400}, {580, 280}, {220, 220, 220}, 3, cv::LINE_AA);
    cv::line(img, {100, 60}, {140, 420}, {200, 200, 200}, 2, cv::LINE_AA);
    cv::circle(img, {200, 220}, 55, {0, 200, 0}, 3, cv::LINE_AA);
    cv::circle(img, {420, 300}, 80, {0, 180, 255}, 4, cv::LINE_AA);
    cv::circle(img, {500, 140}, 35, {255, 100, 100}, -1, cv::LINE_AA);
    cv::Mat noise(h, w, CV_8UC3);
    cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(8));
    return img + noise;
}

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    std::string path = demo::getArg(argc, argv, "--image");
    cv::Mat src = path.empty() ? makeHoughScene() : demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "12_hough");

    cv::Mat gray, edges;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.2);
    cv::Canny(gray, edges, 60, 150);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 60, 50, 10);
    cv::Mat lineVis = src.clone();
    for (const auto& l : lines) {
        cv::line(lineVis, {l[0], l[1]}, {l[2], l[3]}, {0, 255, 255}, 2, cv::LINE_AA);
    }

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1.2, 40, 120, 30, 15, 120);
    cv::Mat circleVis = src.clone();
    for (const auto& c : circles) {
        cv::Point center(cvRound(c[0]), cvRound(c[1]));
        int r = cvRound(c[2]);
        cv::circle(circleVis, center, r, {0, 0, 255}, 2, cv::LINE_AA);
        cv::circle(circleVis, center, 2, {0, 255, 0}, -1);
    }

    std::cout << "lines=" << lines.size() << " circles=" << circles.size() << std::endl;
    demo::save(dir, "12_src.png", src);
    demo::save(dir, "12_edges.png", edges);
    demo::save(dir, "12_lines.png", lineVis);
    demo::save(dir, "12_circles.png", circleVis);

    demo::showIfRequested(argc, argv, "Lines", lineVis);
    demo::showIfRequested(argc, argv, "Circles", circleVis);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 13_template_match\n";
    return 0;
}
