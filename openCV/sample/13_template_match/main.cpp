/**
 * 13_template_match —— 模板匹配（中级）
 * 目标：matchTemplate / minMaxLoc / 多尺度匹配
 */
#include "../common/demo_utils.hpp"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    cv::Mat src = demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "13_template_match");

    // 从原图裁一块当模板，再放到略微变化的场景中
    cv::Rect roi(380, 100, 90, 90);
    roi &= cv::Rect(0, 0, src.cols, src.rows);
    cv::Mat templ = src(roi).clone();

    cv::Mat scene = src.clone();
    // 加一点干扰：多画几个方块
    cv::rectangle(scene, {50, 300, 70, 70}, {100, 100, 255}, -1);

    cv::Mat result;
    cv::matchTemplate(scene, templ, result, cv::TM_CCOEFF_NORMED);
    double minV, maxV;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minV, &maxV, &minLoc, &maxLoc);

    cv::Mat vis = scene.clone();
    cv::rectangle(vis, cv::Rect(maxLoc.x, maxLoc.y, templ.cols, templ.rows), {0, 255, 255}, 2);
    cv::putText(vis, cv::format("score=%.3f", maxV), {maxLoc.x, std::max(20, maxLoc.y - 8)},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 255}, 2);

    // 简单多尺度：缩放模板再匹配，取最高分
    double bestScore = maxV;
    cv::Rect bestBox(maxLoc.x, maxLoc.y, templ.cols, templ.rows);
    for (double s : {0.8, 0.9, 1.1, 1.2}) {
        cv::Mat t2;
        cv::resize(templ, t2, cv::Size(), s, s, cv::INTER_AREA);
        if (t2.cols >= scene.cols || t2.rows >= scene.rows) continue;
        cv::Mat r2;
        cv::matchTemplate(scene, t2, r2, cv::TM_CCOEFF_NORMED);
        double mn, mx;
        cv::Point mnL, mxL;
        cv::minMaxLoc(r2, &mn, &mx, &mnL, &mxL);
        if (mx > bestScore) {
            bestScore = mx;
            bestBox = cv::Rect(mxL.x, mxL.y, t2.cols, t2.rows);
        }
    }
    cv::Mat multiVis = scene.clone();
    cv::rectangle(multiVis, bestBox, {0, 0, 255}, 2);
    cv::putText(multiVis, cv::format("best=%.3f", bestScore), {bestBox.x, std::max(20, bestBox.y - 8)},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 255}, 2);

    // 响应图可视化
    cv::Mat resp;
    cv::normalize(result, resp, 0, 255, cv::NORM_MINMAX);
    resp.convertTo(resp, CV_8U);
    cv::applyColorMap(resp, resp, cv::COLORMAP_JET);

    std::cout << "match score=" << maxV << "  multi-scale best=" << bestScore << std::endl;
    demo::save(dir, "13_template.png", templ);
    demo::save(dir, "13_match.png", vis);
    demo::save(dir, "13_multiscale.png", multiVis);
    demo::save(dir, "13_response.png", resp);

    demo::showIfRequested(argc, argv, "Match", vis);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 14_homography\n";
    return 0;
}
