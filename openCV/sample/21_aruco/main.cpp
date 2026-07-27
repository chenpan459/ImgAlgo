/**
 * 21_aruco —— ArUco 码生成与检测（高级）
 * 目标：drawMarker / detectMarkers / drawDetectedMarkers / estimatePose
 */
#include "../common/demo_utils.hpp"
#include <opencv2/aruco.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    std::string dir = demo::outDir(argc, argv, "21_aruco");

    cv::Ptr<cv::aruco::Dictionary> dict =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

    // 生成若干 marker 贴到画布上，再做仿射扰动后检测
    cv::Mat canvas(520, 720, CV_8UC3, cv::Scalar(240, 240, 240));
    std::vector<int> ids = {0, 1, 2, 3};
    std::vector<cv::Point> origins = {{40, 40}, {400, 50}, {60, 300}, {420, 310}};
    for (size_t i = 0; i < ids.size(); ++i) {
        cv::Mat marker;
        cv::aruco::drawMarker(dict, ids[i], 160, marker, 1);
        cv::cvtColor(marker, marker, cv::COLOR_GRAY2BGR);
        marker.copyTo(canvas(cv::Rect(origins[i].x, origins[i].y, marker.cols, marker.rows)));
    }

    cv::Mat warped;
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(360, 260), 12.0, 0.92);
    M.at<double>(0, 2) += 15;
    cv::warpAffine(canvas, warped, M, canvas.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar(200, 200, 200));

    std::vector<int> detectedIds;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
    cv::aruco::detectMarkers(warped, dict, corners, detectedIds, params);

    cv::Mat vis = warped.clone();
    if (!detectedIds.empty()) {
        cv::aruco::drawDetectedMarkers(vis, corners, detectedIds);
    }

    // 假设针孔相机，估计姿态（演示用虚拟内参）
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 800, 0, warped.cols / 2.0, 0, 800,
                            warped.rows / 2.0, 0, 0, 1);
    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    if (!detectedIds.empty()) {
        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);
        for (size_t i = 0; i < detectedIds.size(); ++i) {
            cv::aruco::drawAxis(vis, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.03);
        }
    }

    std::cout << "detected markers=" << detectedIds.size() << " ids:";
    for (int id : detectedIds) std::cout << " " << id;
    std::cout << std::endl;

    demo::save(dir, "21_board.png", canvas);
    demo::save(dir, "21_warped.png", warped);
    demo::save(dir, "21_detected.png", vis);

    demo::showIfRequested(argc, argv, "ArUco", vis);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 22_kalman_track\n";
    return 0;
}
