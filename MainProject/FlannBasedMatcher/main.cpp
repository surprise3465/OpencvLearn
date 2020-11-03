#include <QCoreApplication>
#include <QThread>
#include <QDebug>
// opencv
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xphoto.hpp"
// opencv_contrib
#include <opencv2/xphoto.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>

#define CVUI_IMPLEMENTATION
#include "../cvui.h"

using namespace cv;
using namespace cvui;
using namespace std;

QString WINDOW_NAME	= "Flann Base Matcher";
QString FOLDERPATH = "D:/workspace/sample/";

int main(int argc, char *argv[])
{
    QString fileName1 = FOLDERPATH + "airplane.tiff";
    QString fileName2 = FOLDERPATH + "plane.tiff";
    int width = 400;
    int height = 300;

    cv::Mat srcMat = cv::imread(fileName1.toStdString());
    cv::Mat srcMat3 = cv::imread(fileName2.toStdString());
    cv::resize(srcMat, srcMat, cv::Size(width, height));
    cv::resize(srcMat3, srcMat3, cv::Size(width, height));


    cv::String windowName = WINDOW_NAME.toStdString();
    cvui::init(windowName);

    cv::Mat windowMat = cv::Mat(cv::Size(srcMat.cols * 2, srcMat.rows * 3),
                                srcMat.type());

    cv::Ptr<cv::xfeatures2d::SIFT> _pSift = cv::xfeatures2d::SiftFeatureDetector::create();
    cv::Ptr<cv::xfeatures2d::SURF> _pSurf = cv::xfeatures2d::SurfFeatureDetector::create();

    cv::Ptr<cv::Feature2D> _pFeature2D;

    int type = 0;
    int k1x = 0;
    int k1y = 0;
    int k2x = 100;
    int k2y = 0;
    int k3x = 100;
    int k3y = 100;
    int k4x = 0;
    int k4y = 100;

    // 定义匹配器
    cv::Ptr<cv::FlannBasedMatcher> pFlannBasedMatcher = cv::FlannBasedMatcher::create();
    // 定义结果存放
    std::vector<cv::DMatch> listDMatch;
    // 存储特征点检测器检测特征后的描述字
    cv::Mat descriptor1;
    cv::Mat descriptor2;

    bool moveFlag = true;  // 移动的标志，不用每次都匹配
    windowMat = cv::Scalar(0, 0, 0);
    while(true)
    {
        cv::Mat mat;
        {
            std::vector<cv::KeyPoint> keyPoints1;
            std::vector<cv::KeyPoint> keyPoints2;

            int typeOld = type;
            int k1xOld = k1x;
            int k1yOld = k1y;
            int k2xOld = k2x;
            int k2yOld = k2y;
            int k3xOld = k3x;
            int k3yOld = k3y;
            int k4xOld = k4x;
            int k4yOld = k4y;

            mat = windowMat(cv::Range(srcMat.rows * 0, srcMat.rows * 1),
                            cv::Range(srcMat.cols * 0, srcMat.cols * 1));
            mat = cv::Scalar(0);


            cvui::trackbar(windowMat, 0 + width * 0, 0 + height * 0, 165, &type, 0, 1);
            cv::String str;
            switch(type)
            {
            case 0:
                str = "sift";
                _pFeature2D = _pSift;
                break;
            case 1:
                str = "surf";
                _pFeature2D = _pSurf;
                break;
            default:
                break;
            }
            cvui::printf(windowMat, width / 2 + width * 0, 20 + height * 0, str.c_str());

            cvui::printf(windowMat, 0 + width * 0, 60 + height * 0, "k1x");
            cvui::trackbar(windowMat, 0 + width * 0, 70 + height * 0, 165, &k1x, 0, 100);
            cvui::printf(windowMat, 0 + width * 0, 120 + height * 0, "k1y");
            cvui::trackbar(windowMat, 0 + width * 0, 130 + height * 0, 165, &k1y, 0, 100);

            cvui::printf(windowMat, width / 2 + width * 0, 60 + height * 0, "k2x");
            cvui::trackbar(windowMat, width / 2 + width * 0, 70 + height * 0, 165, &k2x, 0, 100);
            cvui::printf(windowMat, width / 2 + width * 0, 120 + height * 0, "k2y");
            cvui::trackbar(windowMat, width / 2 + width * 0, 130 + height * 0, 165, &k2y, 0, 100);

            cvui::printf(windowMat, 0 + width * 0, 30 + height * 0 + height / 2, "k3x");
            cvui::trackbar(windowMat, 0 + width * 0, 40 + height * 0 + height / 2, 165, &k3x, 0, 100);
            cvui::printf(windowMat, 0 + width * 0, 90 + height * 0 + height / 2, "k3y");
            cvui::trackbar(windowMat, 0 + width * 0, 100 + height * 0 + height / 2, 165, &k3y, 0, 100);

            cvui::printf(windowMat, width / 2 + width * 0, 30 + height * 0 + height / 2, "k4x");
            cvui::trackbar(windowMat, width / 2 + width * 0, 40 + height * 0 + height / 2, 165, &k4x, 0, 100);
            cvui::printf(windowMat, width / 2 + width * 0, 90 + height * 0 + height / 2, "k4y");
            cvui::trackbar(windowMat, width / 2 + width * 0, 100 + height * 0 + height / 2, 165, &k4y, 0, 100);


            if( k1xOld != k1x || k1yOld != k1y
             || k2xOld != k2x || k2yOld != k2y
             || k3xOld != k3x || k3yOld != k3y
             || k4xOld != k4x || k4yOld != k4y
             || typeOld != type)
            {
                moveFlag = true;
            }

            std::vector<cv::Point2f> srcPoints;
            std::vector<cv::Point2f> dstPoints;

            srcPoints.push_back(cv::Point2f(0.0f, 0.0f));
            srcPoints.push_back(cv::Point2f(srcMat.cols - 1, 0.0f));
            srcPoints.push_back(cv::Point2f(srcMat.cols - 1, srcMat.rows - 1));
            srcPoints.push_back(cv::Point2f(0.0f, srcMat.rows - 1));

            dstPoints.push_back(cv::Point2f(srcMat.cols * k1x / 100.0f, srcMat.rows * k1y / 100.0f));
            dstPoints.push_back(cv::Point2f(srcMat.cols * k2x / 100.0f, srcMat.rows * k2y / 100.0f));
            dstPoints.push_back(cv::Point2f(srcMat.cols * k3x / 100.0f, srcMat.rows * k3y / 100.0f));
            dstPoints.push_back(cv::Point2f(srcMat.cols * k4x / 100.0f, srcMat.rows * k4y / 100.0f));

            cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);
            cv::Mat srcMat2;
            cv::warpPerspective(srcMat3,
                                srcMat2,
                                M,
                                cv::Size(srcMat.cols, srcMat.rows),
                                cv::INTER_LINEAR,
                                cv::BORDER_CONSTANT,
                                cv::Scalar::all(0));

            mat = windowMat(cv::Range(srcMat.rows * 0, srcMat.rows * 1),
                            cv::Range(srcMat.cols * 1, srcMat.cols * 2));
            cv::addWeighted(mat, 0.0f, srcMat2, 1.0f, 0.0f, mat);

            if(moveFlag)
            {
                moveFlag = false;
                //特征点检测
    //           _pSift->detect(srcMat, keyPoints1);
                _pFeature2D->detectAndCompute(srcMat, cv::Mat(), keyPoints1, descriptor1);
                //绘制特征点(关键点)
                cv::Mat resultShowMat;
                cv::drawKeypoints(srcMat,
                                  keyPoints1,
                                  resultShowMat,
                                  cv::Scalar(0, 0, 255),
                                  cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                mat = windowMat(cv::Range(srcMat.rows * 1, srcMat.rows * 2),
                                cv::Range(srcMat.cols * 0, srcMat.cols * 1));
                cv::addWeighted(mat, 0.0f, resultShowMat, 1.0f, 0.0f, mat);

                //特征点检测
    //            _pSift->detect(srcMat2, keyPoints2);
                _pFeature2D->detectAndCompute(srcMat2, cv::Mat(), keyPoints2, descriptor2);
                //绘制特征点(关键点)
                cv::Mat resultShowMat2;
                cv::drawKeypoints(srcMat2,
                                  keyPoints2,
                                  resultShowMat2,
                                  cv::Scalar(0, 0, 255),
                                  cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                mat = windowMat(cv::Range(srcMat.rows * 1, srcMat.rows * 2),
                                cv::Range(srcMat.cols * 1, srcMat.cols * 2));
                cv::addWeighted(mat, 0.0f, resultShowMat2, 1.0f, 0.0f, mat);

                // FlannBasedMatcher最近邻匹配
                pFlannBasedMatcher->match(descriptor1, descriptor2, listDMatch);
                // drawMatch绘制出来，并排显示了，高度一样，宽度累加（因为两个宽度相同，所以是两倍了）
                cv::Mat matchesMat;
                cv::drawMatches(srcMat,
                                keyPoints1,
                                srcMat2,
                                keyPoints2,
                                listDMatch,
                                matchesMat);

                mat = windowMat(cv::Range(srcMat.rows * 2, srcMat.rows * 3),
                                cv::Range(srcMat.cols * 0, srcMat.cols * 2));
                cv::addWeighted(mat, 0.0f, matchesMat, 1.0f, 0.0f, mat);
            }
        }
        cv::imshow(windowName, windowMat);
        // 更新
        cvui::update();
        // 显示
        // esc键退出
        if(cv::waitKey(25) == 27)
        {
            break;
        }
    }
}
