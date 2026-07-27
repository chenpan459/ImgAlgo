/**
 * 04_drawing —— 绘图 API
 * 目标：line / rectangle / circle / ellipse / polylines / putText
 */
#include "../common/demo_utils.hpp"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    std::string dir = demo::outDir(argc, argv, "04_drawing");

    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(20, 20, 20));

    cv::line(img, {50, 50}, {300, 120}, {0, 255, 255}, 2, cv::LINE_AA);
    cv::rectangle(img, cv::Rect(80, 160, 160, 100), {0, 0, 255}, 2);
    cv::circle(img, {420, 200}, 60, {0, 255, 0}, -1, cv::LINE_AA);
    cv::ellipse(img, {420, 360}, cv::Size(90, 40), 30, 0, 360, {255, 128, 0}, 2, cv::LINE_AA);

    std::vector<cv::Point> pts = {{60, 400}, {140, 300}, {220, 420}, {60, 400}};
    cv::polylines(img, pts, false, {255, 0, 255}, 2, cv::LINE_AA);

    cv::putText(img, "OpenCV Drawing", {180, 50}, cv::FONT_HERSHEY_SIMPLEX, 1.1,
                {255, 255, 255}, 2, cv::LINE_AA);
    cv::arrowedLine(img, {500, 80}, {580, 140}, {200, 200, 255}, 2, cv::LINE_AA, 0, 0.25);

    demo::save(dir, "04_drawing.png", img);
    demo::showIfRequested(argc, argv, "Drawing", img);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 05_filter\n";
    return 0;
}
