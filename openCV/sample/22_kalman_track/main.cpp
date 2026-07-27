/**
 * 22_kalman_track —— 卡尔曼滤波跟踪（高级）
 * 目标：模拟弹跳点 + KalmanFilter 预测/校正，对比观测与估计
 */
#include "../common/demo_utils.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0], "[--frames=60]");
    std::string dir = demo::outDir(argc, argv, "22_kalman_track");
    int nFrames = std::stoi(demo::getArg(argc, argv, "--frames", "60"));

    cv::KalmanFilter kf(4, 2, 0);
    // state: [x, y, vx, vy]
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
    cv::setIdentity(kf.measurementMatrix);
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

    float x = 80.f, y = 80.f, vx = 7.f, vy = 5.f;
    const int W = 640, H = 480;
    kf.statePost = (cv::Mat_<float>(4, 1) << x, y, vx, vy);

    cv::Mat trail(H, W, CV_8UC3, cv::Scalar(20, 20, 20));
    std::vector<cv::Point> measPath, estPath;

    for (int i = 0; i < nFrames; ++i) {
        // 真值运动 + 弹边
        x += vx;
        y += vy;
        if (x < 20 || x > W - 20) vx = -vx;
        if (y < 20 || y > H - 20) vy = -vy;

        // 带噪声观测
        cv::Mat meas = (cv::Mat_<float>(2, 1) << x + 4.f * ((rand() % 100) / 100.f - 0.5f),
                        y + 4.f * ((rand() % 100) / 100.f - 0.5f));

        cv::Mat pred = kf.predict();
        cv::Mat est = kf.correct(meas);

        cv::Point pm(cvRound(meas.at<float>(0)), cvRound(meas.at<float>(1)));
        cv::Point pe(cvRound(est.at<float>(0)), cvRound(est.at<float>(1)));
        cv::Point pp(cvRound(pred.at<float>(0)), cvRound(pred.at<float>(1)));
        measPath.push_back(pm);
        estPath.push_back(pe);

        cv::Mat frame = trail.clone();
        for (size_t k = 1; k < measPath.size(); ++k) {
            cv::line(frame, measPath[k - 1], measPath[k], {80, 80, 255}, 1, cv::LINE_AA);
            cv::line(frame, estPath[k - 1], estPath[k], {80, 255, 80}, 2, cv::LINE_AA);
        }
        cv::circle(frame, pm, 5, {0, 0, 255}, -1);
        cv::circle(frame, pe, 5, {0, 255, 0}, -1);
        cv::circle(frame, pp, 4, {255, 255, 0}, 1);
        cv::putText(frame, "red=meas green=est yellow=pred", {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    {220, 220, 220}, 1);

        if (i == 0 || i == nFrames / 2 || i == nFrames - 1) {
            demo::save(dir, cv::format("22_frame_%02d.png", i), frame);
        }
        if (i == nFrames - 1) {
            demo::save(dir, "22_final.png", frame);
            demo::showIfRequested(argc, argv, "Kalman", frame);
        }
    }

    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 23_document_scan\n";
    return 0;
}
