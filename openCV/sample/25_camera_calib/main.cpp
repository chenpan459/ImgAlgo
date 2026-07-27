/**
 * 25_camera_calib —— 相机标定（高级）
 * 目标：合成多视角棋盘格 → findChessboardCorners → calibrateCamera
 */
#include "../common/demo_utils.hpp"
#include <cmath>
#include <iostream>
#include <vector>

static cv::Mat renderChessboard(const cv::Size& boardSize, int squarePx, double yawDeg, double pitchDeg,
                                cv::Size imgSize) {
    // 生成正面棋盘，再做单应性投影
    int w = (boardSize.width + 1) * squarePx;
    int h = (boardSize.height + 1) * squarePx;
    cv::Mat board(h, w, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int r = 0; r < boardSize.height + 1; ++r) {
        for (int c = 0; c < boardSize.width + 1; ++c) {
            if ((r + c) % 2 == 0) {
                cv::rectangle(board, cv::Rect(c * squarePx, r * squarePx, squarePx, squarePx),
                              cv::Scalar(0, 0, 0), -1);
            }
        }
    }

    std::vector<cv::Point2f> src = {{0, 0},
                                    {static_cast<float>(w - 1), 0},
                                    {static_cast<float>(w - 1), static_cast<float>(h - 1)},
                                    {0, static_cast<float>(h - 1)}};
    float cx = imgSize.width * 0.5f, cy = imgSize.height * 0.5f;
    float scale = 0.55f;
    float dx = 40.f * static_cast<float>(std::sin(yawDeg * CV_PI / 180.0));
    float dy = 30.f * static_cast<float>(std::sin(pitchDeg * CV_PI / 180.0));
    std::vector<cv::Point2f> dst = {
        {cx - w * scale / 2 + dx, cy - h * scale / 2 + dy},
        {cx + w * scale / 2 + dx * 0.3f, cy - h * scale / 2 - dy * 0.2f},
        {cx + w * scale / 2 - dx * 0.2f, cy + h * scale / 2 + dy * 0.4f},
        {cx - w * scale / 2 - dx * 0.4f, cy + h * scale / 2 - dy * 0.1f}};
    cv::Mat H = cv::getPerspectiveTransform(src, dst);
    cv::Mat out(imgSize, CV_8UC3, cv::Scalar(180, 180, 180));
    cv::warpPerspective(board, out, H, imgSize, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
    return out;
}

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    std::string dir = demo::outDir(argc, argv, "25_camera_calib");

    const cv::Size pattern(9, 6);  // 内角点数
    const float squareSize = 1.0f;
    cv::Size imgSize(640, 480);

    std::vector<std::vector<cv::Point3f>> objPoints;
    std::vector<std::vector<cv::Point2f>> imgPoints;
    std::vector<cv::Point3f> obj;
    for (int i = 0; i < pattern.height; ++i)
        for (int j = 0; j < pattern.width; ++j) obj.emplace_back(j * squareSize, i * squareSize, 0);

    int found = 0;
    for (int k = 0; k < 12; ++k) {
        double yaw = -20 + k * 4.0;
        double pitch = -10 + (k % 5) * 5.0;
        cv::Mat img = renderChessboard(pattern, 40, yaw, pitch, imgSize);
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;
        bool ok = cv::findChessboardCorners(
            gray, pattern, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (ok) {
            cv::cornerSubPix(gray, corners, {11, 11}, {-1, -1},
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
            objPoints.push_back(obj);
            imgPoints.push_back(corners);
            cv::drawChessboardCorners(img, pattern, corners, ok);
            ++found;
            if (found <= 3) demo::save(dir, cv::format("25_view_%02d.png", found), img);
        }
    }

    std::cout << "chessboard views found=" << found << std::endl;
    if (found < 3) {
        std::cerr << "Not enough views for calibration\n";
        return 1;
    }

    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(objPoints, imgPoints, imgSize, cameraMatrix, distCoeffs, rvecs, tvecs);

    std::cout << "RMS reprojection error=" << rms << std::endl;
    std::cout << "cameraMatrix=\n" << cameraMatrix << std::endl;
    std::cout << "distCoeffs=" << distCoeffs.t() << std::endl;

    // 保存去畸变示例
    cv::Mat sample = renderChessboard(pattern, 40, 8, 6, imgSize);
    cv::Mat undist;
    cv::undistort(sample, undist, cameraMatrix, distCoeffs);
    demo::save(dir, "25_distorted.png", sample);
    demo::save(dir, "25_undistorted.png", undist);

    cv::FileStorage fs(dir + "/25_calib.yml", cv::FileStorage::WRITE);
    fs << "rms" << rms << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;
    fs.release();
    std::cout << "Saved calib: " << dir << "/25_calib.yml\n";

    demo::showIfRequested(argc, argv, "Undistort", undist);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 26_inpaint_clone\n";
    return 0;
}
