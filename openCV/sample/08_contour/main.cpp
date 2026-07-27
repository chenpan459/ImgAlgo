/**
 * 08_contour —— 轮廓检测与几何特征
 * 目标：findContours / boundingRect / minAreaRect / moments / approxPolyDP
 */
#include "../common/demo_utils.hpp"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    cv::Mat src = demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "08_contour");

    cv::Mat gray, bin;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.0);
    cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(bin, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat vis = src.clone();
    int kept = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area < 200) continue;
        ++kept;

        cv::drawContours(vis, contours, static_cast<int>(i), cv::Scalar(0, 255, 255), 2);

        cv::Rect br = cv::boundingRect(contours[i]);
        cv::rectangle(vis, br, cv::Scalar(0, 0, 255), 1);

        cv::RotatedRect rr = cv::minAreaRect(contours[i]);
        cv::Point2f pts[4];
        rr.points(pts);
        for (int k = 0; k < 4; ++k) {
            cv::line(vis, pts[k], pts[(k + 1) % 4], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        }

        std::vector<cv::Point> approx;
        double peri = cv::arcLength(contours[i], true);
        cv::approxPolyDP(contours[i], approx, 0.02 * peri, true);

        cv::Moments m = cv::moments(contours[i]);
        if (m.m00 > 1e-5) {
            int cx = static_cast<int>(m.m10 / m.m00);
            int cy = static_cast<int>(m.m01 / m.m00);
            cv::circle(vis, {cx, cy}, 3, {0, 255, 0}, -1);
            cv::putText(vis, "n=" + std::to_string(approx.size()), {cx + 5, cy},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
        }
    }

    std::cout << "contours=" << contours.size() << " kept(area>=200)=" << kept << std::endl;
    demo::save(dir, "08_bin.png", bin);
    demo::save(dir, "08_contours.png", vis);
    demo::showIfRequested(argc, argv, "Contours", vis);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 09_geometry\n";
    return 0;
}
