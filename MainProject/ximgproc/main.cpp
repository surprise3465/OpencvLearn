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

#define CVUI_IMPLEMENTATION
#include "../cvui.h"

using namespace cv;
using namespace cvui;
using namespace std;
 using namespace cv::ximgproc;
#define WINDOW_NAME	"ximgproc"
#define FILEPATH "D:/workspace/sample/lena(1).tiff"

#include <opencv2/ximgproc/edge_filter.hpp>

int main(int argc, char *argv[])
{
    QString fileName1 = FILEPATH;
    cv::Mat srcMat = cv::imread(fileName1.toStdString());

    int width = 480;
    int height = 320;
    cv::resize(srcMat, srcMat, cv::Size(width, height));

    cv::String windowName = QString(WINDOW_NAME).toStdString();
    cvui::init(windowName);

    cv::Mat windowMat = cv::Mat(cv::Size(srcMat.cols * 2, srcMat.rows * 2),
                                srcMat.type());

    int sigmaS = 160;
    int sigmaR = 2;

    while(true)
    {
        windowMat = cv::Scalar(0, 0, 0);
        // 原图先copy到左边
        cv::Mat leftMat = windowMat(cv::Range(0, srcMat.rows),
                                    cv::Range(0, srcMat.cols));
        cv::addWeighted(leftMat, 0.0f, srcMat, 1.0f, 0.0f, leftMat);

        cv::Mat mat;
        cv::Mat dstMat;

        cvui::printf(windowMat, 75 + width * 1, 40, "sigmaS");
        cvui::trackbar(windowMat, 75 + width * 1, 50, 165, &sigmaS, 101, 10000);
        cvui::printf(windowMat, 75 + width * 1, 90, "sigmaR");
        cvui::trackbar(windowMat, 75 + width * 1, 100, 165, &sigmaR, 1, 100);

        {
            // 使用自适应流形应用高维滤波。
            cv::Ptr<cv::ximgproc::AdaptiveManifoldFilter> pAdaptiveManifoldFilter = cv::ximgproc::createAMFilter(sigmaS/100.0f, sigmaR/100.0f, true);
            pAdaptiveManifoldFilter->filter(srcMat, dstMat);
            // copy到左下
            mat = windowMat(cv::Range(srcMat.rows, srcMat.rows * 2),
                                      cv::Range(srcMat.cols * 0, srcMat.cols * 1));
            cv::addWeighted(mat, 0.0f, dstMat, 1.0f, 0.0f, mat);
        }

        {
            // 使用自适应流形应用高维滤波。
            cv::Ptr<cv::ximgproc::AdaptiveManifoldFilter> pAdaptiveManifoldFilter = cv::ximgproc::createAMFilter(sigmaS/100.0f, sigmaR/100.0f, false);
            pAdaptiveManifoldFilter->filter(srcMat, dstMat);
            // copy到左下
            mat = windowMat(cv::Range(srcMat.rows, srcMat.rows * 2),
                                      cv::Range(srcMat.cols * 1, srcMat.cols * 2));
            cv::addWeighted(mat, 0.0f, dstMat, 1.0f, 0.0f, mat);
        }

        // 更新
        cvui::update();
        // 显示
        cv::imshow(windowName, windowMat);
        // esc键退出
        if(cv::waitKey(25) == 27)
        {
            break;
        }
    }

    return 0;
}
