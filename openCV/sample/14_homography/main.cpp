/**
 * 14_homography —— 特征 + 单应性对齐（中级）
 * 目标：ORB匹配 → findHomography(RANSAC) → warpPerspective
 */
#include "../common/demo_utils.hpp"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    cv::Mat img1 = demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "14_homography");

    // 模拟第二张图：透视扰动
    std::vector<cv::Point2f> srcQuad = {
        {0.f, 0.f},
        {static_cast<float>(img1.cols - 1), 0.f},
        {static_cast<float>(img1.cols - 1), static_cast<float>(img1.rows - 1)},
        {0.f, static_cast<float>(img1.rows - 1)}};
    std::vector<cv::Point2f> dstQuad = {
        {40.f, 30.f},
        {static_cast<float>(img1.cols) - 20.f, 10.f},
        {static_cast<float>(img1.cols) - 50.f, static_cast<float>(img1.rows) - 25.f},
        {25.f, static_cast<float>(img1.rows) - 40.f}};
    cv::Mat Hgt = cv::getPerspectiveTransform(srcQuad, dstQuad);
    cv::Mat img2;
    cv::warpPerspective(img1, img2, Hgt, img1.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    auto orb = cv::ORB::create(800);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, des1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, des2);

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(des1, des2, knn, 2);

    std::vector<cv::DMatch> good;
    for (const auto& m : knn) {
        if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance) good.push_back(m[0]);
    }

    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : good) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }

    cv::Mat H, mask;
    cv::Mat aligned, matchVis;
    if (pts1.size() >= 4) {
        H = cv::findHomography(pts2, pts1, cv::RANSAC, 3.0, mask);
        if (!H.empty()) {
            cv::warpPerspective(img2, aligned, H, img1.size());
        }
    }

    cv::drawMatches(img1, kp1, img2, kp2, good, matchVis, cv::Scalar(0, 255, 255),
                    cv::Scalar(0, 0, 255), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    int inliers = mask.empty() ? 0 : cv::countNonZero(mask);
    std::cout << "good matches=" << good.size() << " inliers=" << inliers << std::endl;

    // 叠加对比：对齐后与原图差异
    cv::Mat diff;
    if (!aligned.empty()) {
        cv::absdiff(img1, aligned, diff);
    }

    demo::save(dir, "14_img1.png", img1);
    demo::save(dir, "14_img2.png", img2);
    demo::save(dir, "14_matches.png", matchVis);
    if (!aligned.empty()) demo::save(dir, "14_aligned.png", aligned);
    if (!diff.empty()) demo::save(dir, "14_diff.png", diff);

    demo::showIfRequested(argc, argv, "Aligned", aligned.empty() ? matchVis : aligned);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 15_connected_components\n";
    return 0;
}
