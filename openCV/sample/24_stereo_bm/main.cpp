/**
 * 24_stereo_bm —— 双目立体匹配（高级）
 * 目标：合成左右目视差图 + StereoBM 求 disparity / 伪彩色深度
 */
#include "../common/demo_utils.hpp"
#include <iostream>

static void makeStereoPair(cv::Mat& left, cv::Mat& right, int w = 640, int h = 480) {
    left = cv::Mat(h, w, CV_8UC3, cv::Scalar(40, 40, 40));
    // 不同深度的物体（水平偏移模拟视差）
    struct Obj {
        cv::Rect r;
        cv::Scalar color;
        int disparity;
    };
    std::vector<Obj> objs = {
        {{80, 120, 100, 80}, {0, 0, 220}, 24},
        {{280, 160, 120, 100}, {0, 200, 0}, 16},
        {{450, 200, 90, 90}, {220, 180, 0}, 10},
        {{150, 320, 200, 60}, {200, 80, 200}, 6},
    };
    for (const auto& o : objs) {
        cv::rectangle(left, o.r, o.color, -1);
    }
    // 右图：向左平移 disparity
    right = cv::Mat(h, w, CV_8UC3, cv::Scalar(40, 40, 40));
    for (const auto& o : objs) {
        cv::Rect rr = o.r;
        rr.x -= o.disparity;
        rr &= cv::Rect(0, 0, w, h);
        if (rr.area() > 0) cv::rectangle(right, rr, o.color, -1);
    }
    cv::Mat n1(h, w, CV_8UC3), n2(h, w, CV_8UC3);
    cv::randn(n1, cv::Scalar::all(0), cv::Scalar::all(4));
    cv::randn(n2, cv::Scalar::all(0), cv::Scalar::all(4));
    left += n1;
    right += n2;
}

int main(int argc, char** argv) {
    demo::printHelpHint(argv[0]);
    std::string dir = demo::outDir(argc, argv, "24_stereo_bm");

    cv::Mat left, right;
    makeStereoPair(left, right);

    cv::Mat gL, gR, disp16, disp8, dispColor;
    cv::cvtColor(left, gL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, gR, cv::COLOR_BGR2GRAY);

    auto bm = cv::StereoBM::create(64, 15);
    bm->setPreFilterCap(31);
    bm->setMinDisparity(0);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(10);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->compute(gL, gR, disp16);

    // StereoBM 输出为 16SC1，真实视差 = value/16
    disp16.convertTo(disp8, CV_8U, 255.0 / (64 * 16.0));
    cv::applyColorMap(disp8, dispColor, cv::COLORMAP_JET);

    demo::save(dir, "24_left.png", left);
    demo::save(dir, "24_right.png", right);
    demo::save(dir, "24_disp.png", disp8);
    demo::save(dir, "24_disp_color.png", dispColor);

    demo::showIfRequested(argc, argv, "Disparity", dispColor);
    demo::waitIfShown(argc, argv);
    std::cout << "Done. Next: 25_camera_calib\n";
    return 0;
}
