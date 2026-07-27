/**
 * 10_feature_match —— ORB 特征与匹配
 * 目标：ORB detectAndCompute / BFMatcher / drawMatches
 * 进阶可看 DiffImg（SSIM / pHash / 更完整比较）
 */
#include "../common/demo_utils.hpp"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    cv::Mat img1 = demo::loadOrToy(argc, argv);
    std::string dir = demo::outDir(argc, argv, "10_feature_match");

    // 人为制造第二张：旋转 + 平移，模拟“相似但不相同”
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(img1.cols / 2.f, img1.rows / 2.f), 15.0, 0.95);
    M.at<double>(0, 2) += 20;
    M.at<double>(1, 2) += -10;
    cv::Mat img2;
    cv::warpAffine(img1, img2, M, img1.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);

    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    orb->detectAndCompute(img1, cv::noArray(), kp1, des1);
    orb->detectAndCompute(img2, cv::noArray(), kp2, des2);

    if (des1.empty() || des2.empty()) {
        std::cerr << "No descriptors found\n";
        return 1;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(des1, des2, matches);

    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });
    const size_t keep = std::min<size_t>(50, matches.size());
    matches.resize(keep);

    cv::Mat kpVis1, matchVis;
    cv::drawKeypoints(img1, kp1, kpVis1, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
    cv::drawMatches(img1, kp1, img2, kp2, matches, matchVis, cv::Scalar(0, 255, 255),
                    cv::Scalar(0, 0, 255), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    std::cout << "keypoints: " << kp1.size() << " / " << kp2.size()
              << "  good matches: " << matches.size() << std::endl;

    demo::save(dir, "10_img1.png", img1);
    demo::save(dir, "10_img2.png", img2);
    demo::save(dir, "10_keypoints.png", kpVis1);
    demo::save(dir, "10_matches.png", matchVis);

    demo::showIfRequested(argc, argv, "Matches", matchVis);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Advanced: ../DiffImg (image similarity)\n";
    return 0;
}
