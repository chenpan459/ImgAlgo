#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace cv;
using namespace std;

class ImageComparator {
public:
    // 计算结构相似性指数 (SSIM)
    double calculateSSIM(const Mat& img1, const Mat& img2) {
        const double C1 = 6.5025, C2 = 58.5225;
        
        Mat img1_float, img2_float;
        img1.convertTo(img1_float, CV_32F);
        img2.convertTo(img2_float, CV_32F);
        
        Mat img1_sq = img1_float.mul(img1_float);
        Mat img2_sq = img2_float.mul(img2_float);
        Mat img1_img2 = img1_float.mul(img2_float);
        
        Mat mu1, mu2;
        GaussianBlur(img1_float, mu1, Size(11, 11), 1.5);
        GaussianBlur(img2_float, mu2, Size(11, 11), 1.5);
        
        Mat mu1_sq = mu1.mul(mu1);
        Mat mu2_sq = mu2.mul(mu2);
        Mat mu1_mu2 = mu1.mul(mu2);
        
        Mat sigma1_sq, sigma2_sq, sigma12;
        GaussianBlur(img1_sq, sigma1_sq, Size(11, 11), 1.5);
        sigma1_sq -= mu1_sq;
        
        GaussianBlur(img2_sq, sigma2_sq, Size(11, 11), 1.5);
        sigma2_sq -= mu2_sq;
        
        GaussianBlur(img1_img2, sigma12, Size(11, 11), 1.5);
        sigma12 -= mu1_mu2;
        
        Mat ssim_map = ((2 * mu1_mu2 + C1).mul(2 * sigma12 + C2)) / 
                       ((mu1_sq + mu2_sq + C1).mul(sigma1_sq + sigma2_sq + C2));
        
        Scalar mean_ssim = mean(ssim_map);
        return (mean_ssim[0] + mean_ssim[1] + mean_ssim[2]) / 3.0;
    }
    
    // 计算感知哈希 (pHash)
    string calculatePerceptualHash(const Mat& img) {
        Mat resized, gray, dct;
        
        // 调整大小到8x8
        resize(img, resized, Size(8, 8));
        
        // 转换为灰度图
        if (resized.channels() == 3) {
            cvtColor(resized, gray, COLOR_BGR2GRAY);
        } else {
            gray = resized;
        }
        
        // 转换为浮点数并计算DCT
        Mat floatImg;
        gray.convertTo(floatImg, CV_32F);
        dct(floatImg, dct);
        
        // 取左上角8x8的DCT系数
        Mat dctLowFreq = dct(Rect(0, 0, 8, 8));
        
        // 计算平均值（排除DC系数）
        Scalar meanVal = mean(dctLowFreq(Rect(1, 1, 7, 7)));
        double avg = meanVal[0];
        
        // 生成哈希
        string hash;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (i == 0 && j == 0) continue; // 跳过DC系数
                hash += (dctLowFreq.at<float>(i, j) > avg) ? "1" : "0";
            }
        }
        
        return hash;
    }
    
    // 计算汉明距离
    int hammingDistance(const string& hash1, const string& hash2) {
        if (hash1.length() != hash2.length()) {
            return -1;
        }
        
        int distance = 0;
        for (size_t i = 0; i < hash1.length(); i++) {
            if (hash1[i] != hash2[i]) {
                distance++;
            }
        }
        return distance;
    }
    
    // 使用特征点匹配
    double featureMatch(const Mat& img1, const Mat& img2) {
        Mat gray1, gray2;
        
        if (img1.channels() == 3) {
            cvtColor(img1, gray1, COLOR_BGR2GRAY);
        } else {
            gray1 = img1;
        }
        
        if (img2.channels() == 3) {
            cvtColor(img2, gray2, COLOR_BGR2GRAY);
        } else {
            gray2 = img2;
        }
        
        // 使用ORB特征检测器
        Ptr<ORB> detector = ORB::create();
        vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
        
        detector->detectAndCompute(gray1, noArray(), keypoints1, descriptors1);
        detector->detectAndCompute(gray2, noArray(), keypoints2, descriptors2);
        
        if (descriptors1.empty() || descriptors2.empty()) {
            return 0.0;
        }
        
        // 使用BFMatcher进行匹配
        BFMatcher matcher(NORM_HAMMING);
        vector<vector<DMatch>> knnMatches;
        matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);
        
        // 应用Lowe's ratio test
        vector<DMatch> goodMatches;
        const float ratio_thresh = 0.75f;
        for (size_t i = 0; i < knnMatches.size(); i++) {
            if (knnMatches[i].size() == 2 && 
                knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
                goodMatches.push_back(knnMatches[i][0]);
            }
        }
        
        // 计算匹配率
        int maxKeypoints = max(keypoints1.size(), keypoints2.size());
        if (maxKeypoints == 0) return 0.0;
        
        return (double)goodMatches.size() / maxKeypoints;
    }
    
    // 计算直方图相似度
    double histogramSimilarity(const Mat& img1, const Mat& img2) {
        Mat hist1, hist2;
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        
        vector<Mat> bgr_planes1, bgr_planes2;
        split(img1, bgr_planes1);
        split(img2, bgr_planes2);
        
        double similarity = 0.0;
        int channels = min(img1.channels(), img2.channels());
        
        for (int i = 0; i < channels; i++) {
            calcHist(&bgr_planes1[i], 1, 0, Mat(), hist1, 1, &histSize, &histRange);
            calcHist(&bgr_planes2[i], 1, 0, Mat(), hist2, 1, &histSize, &histRange);
            
            normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
            normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());
            
            similarity += compareHist(hist1, hist2, HISTCMP_CORREL);
        }
        
        return similarity / channels;
    }
    
    // 综合比较方法
    struct ComparisonResult {
        double ssim;
        double perceptualHash;
        double featureMatch;
        double histogram;
        double overall;
        bool isSame;
    };
    
    ComparisonResult compareImages(const string& img1Path, const string& img2Path) {
        ComparisonResult result;
        
        Mat img1 = imread(img1Path);
        Mat img2 = imread(img2Path);
        
        if (img1.empty() || img2.empty()) {
            cerr << "错误: 无法读取图像文件" << endl;
            result.overall = -1.0;
            result.isSame = false;
            return result;
        }
        
        // 统一图像尺寸
        if (img1.size() != img2.size()) {
            resize(img2, img2, img1.size());
        }
        
        // 计算SSIM
        result.ssim = calculateSSIM(img1, img2);
        
        // 计算感知哈希
        string hash1 = calculatePerceptualHash(img1);
        string hash2 = calculatePerceptualHash(img2);
        int hammingDist = hammingDistance(hash1, hash2);
        result.perceptualHash = 1.0 - (double)hammingDist / hash1.length();
        
        // 特征匹配
        result.featureMatch = featureMatch(img1, img2);
        
        // 直方图相似度
        result.histogram = histogramSimilarity(img1, img2);
        
        // 综合评分（加权平均）
        result.overall = (result.ssim * 0.4 + 
                         result.perceptualHash * 0.3 + 
                         result.featureMatch * 0.2 + 
                         result.histogram * 0.1);
        
        // 判断是否为同一画面（阈值可调整）
        result.isSame = result.overall > 0.85;
        
        return result;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "用法: " << argv[0] << " <图片1路径> <图片2路径>" << endl;
        cout << "示例: " << argv[0] << " image1.jpg image2.jpg" << endl;
        return -1;
    }
    
    ImageComparator comparator;
    ImageComparator::ComparisonResult result = comparator.compareImages(argv[1], argv[2]);
    
    if (result.overall < 0) {
        return -1;
    }
    
    cout << "\n========== 图像比较结果 ==========" << endl;
    cout << "SSIM相似度:        " << fixed << setprecision(4) << result.ssim << endl;
    cout << "感知哈希相似度:    " << fixed << setprecision(4) << result.perceptualHash << endl;
    cout << "特征匹配相似度:    " << fixed << setprecision(4) << result.featureMatch << endl;
    cout << "直方图相似度:      " << fixed << setprecision(4) << result.histogram << endl;
    cout << "-----------------------------------" << endl;
    cout << "综合相似度:        " << fixed << setprecision(4) << result.overall << endl;
    cout << "判断结果:          " << (result.isSame ? "是同一画面" : "不是同一画面") << endl;
    cout << "===================================" << endl;
    
    return result.isSame ? 0 : 1;
}
