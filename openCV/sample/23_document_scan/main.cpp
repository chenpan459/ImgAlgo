/**
 * 23_document_scan —— 文档扫描流水线（高级综合）
 * 目标：边缘 → 最大四边形轮廓 → 透视拉正（A4 比例）
 */
#include "../common/demo_utils.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

static cv::Mat makeDocScene(int w = 640, int h = 480) {
    cv::Mat bg(h, w, CV_8UC3, cv::Scalar(60, 70, 80));
    // 透视倾斜的“纸张”
    std::vector<cv::Point> quad = {{120, 90}, {520, 70}, {560, 400}, {90, 390}};
    cv::fillConvexPoly(bg, quad, cv::Scalar(245, 245, 245));
    // 纸上文字线条
    for (int i = 0; i < 8; ++i) {
        int y = 140 + i * 28;
        cv::line(bg, {160, y}, {500, y - 6}, {30, 30, 30}, 2, cv::LINE_AA);
    }
    cv::rectangle(bg, {180, 300, 120, 50}, {0, 0, 200}, -1);
    cv::putText(bg, "DOC", {200, 335}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {255, 255, 255}, 2);
    cv::Mat noise(h, w, CV_8UC3);
    cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(5));
    return bg + noise;
}

static std::vector<cv::Point2f> orderQuad(const std::vector<cv::Point>& pts) {
    CV_Assert(pts.size() == 4);
    std::vector<cv::Point2f> p(4);
    for (int i = 0; i < 4; ++i) p[i] = pts[i];
    std::sort(p.begin(), p.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y == b.y ? a.x < b.x : a.y < b.y;
    });
    std::vector<cv::Point2f> top = {p[0], p[1]}, bot = {p[2], p[3]};
    if (top[0].x > top[1].x) std::swap(top[0], top[1]);
    if (bot[0].x > bot[1].x) std::swap(bot[0], bot[1]);
    // tl, tr, br, bl
    return {top[0], top[1], bot[1], bot[0]};
}

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    std::string path = demo::getArg(argc, argv, "--image");
    cv::Mat src = path.empty() ? makeDocScene() : demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "23_document_scan");

    cv::Mat gray, blur, edges;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
    cv::Canny(blur, edges, 50, 150);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    std::sort(contours.begin(), contours.end(),
              [](const auto& a, const auto& b) { return cv::contourArea(a) > cv::contourArea(b); });

    std::vector<cv::Point> docApprox;
    for (const auto& c : contours) {
        std::vector<cv::Point> approx;
        double peri = cv::arcLength(c, true);
        cv::approxPolyDP(c, approx, 0.02 * peri, true);
        if (approx.size() == 4 && cv::contourArea(approx) > 10000 && cv::isContourConvex(approx)) {
            docApprox = approx;
            break;
        }
    }

    cv::Mat marked = src.clone();
    cv::Mat warped;
    if (docApprox.size() == 4) {
        auto ordered = orderQuad(docApprox);
        for (int i = 0; i < 4; ++i) {
            cv::line(marked, ordered[i], ordered[(i + 1) % 4], {0, 255, 255}, 2, cv::LINE_AA);
            cv::circle(marked, ordered[i], 5, {0, 0, 255}, -1);
        }
        std::vector<cv::Point2f> dst = {{0, 0}, {500, 0}, {500, 700}, {0, 700}};
        cv::Mat H = cv::getPerspectiveTransform(ordered, dst);
        cv::warpPerspective(src, warped, H, cv::Size(500, 700));
    } else {
        std::cerr << "No quadrilateral document found; saved edges only\n";
    }

    demo::save(dir, "23_src.png", src);
    demo::save(dir, "23_edges.png", edges);
    demo::save(dir, "23_quad.png", marked);
    if (!warped.empty()) demo::save(dir, "23_warped.png", warped);

    demo::showIfRequested(argc, argv, "Scan", warped.empty() ? marked : warped);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 24_stereo_bm\n";
    return 0;
}
