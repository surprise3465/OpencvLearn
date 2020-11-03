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

QString WINDOW_NAME	= "Flann Base Matcher";
QString FOLDERPATH = "D:/workspace/sample/";

int main(int argc, char *argv[])
{

    cv::Mat srcMat = cv::imread((FOLDERPATH + "lena(1).tiff").toStdString());

    int width = 300;
    int height = 200;
    cv::resize(srcMat, srcMat, cv::Size(width, height));

    cv::String windowName = WINDOW_NAME.toStdString();
    cvui::init(windowName);

    cv::Mat windowMat = cv::Mat(cv::Size(srcMat.cols * 3, srcMat.rows * 4),
                                srcMat.type());

    int x = 100;
    int y = 100;

    int b = 255;
    int g = 0;
    int r = 0;

    int loB = 5;
    int loG = 5;
    int loR = 5;

    int upB = 5;
    int upG = 5;
    int upR = 5;

    while(true)
    {
        windowMat = cv::Scalar(0, 0, 0);
        // 原图先copy到左边
        cv::Mat leftMat = windowMat(cv::Range(0, srcMat.rows),
                                    cv::Range(0, srcMat.cols));
        cv::addWeighted(leftMat, 1.0f, srcMat, 1.0f, 0.0f, leftMat);

        // 选取原图坐标点
        cvui::printf(windowMat, width * 1 + 50, 30 + height * 0, "x");
        cvui::trackbar(windowMat, width * 1 + 50, 40 + height * 0, 200, &x, 0, width);

        cvui::printf(windowMat, width * 1 + 50, 90 + height * 0, "y");
        cvui::trackbar(windowMat, width * 1 + 50, 100 + height * 0, 200, &y, 0, height);

        // 修改的新颜色
        cvui::printf(windowMat, width * 2 + 50, 0 + height * 0, "b");
        cvui::trackbar(windowMat, width * 2 + 50, 10 + height * 0, 200, &b, 0, 255);

        cvui::printf(windowMat, width * 2 + 50, 60 + height * 0, "g");
        cvui::trackbar(windowMat, width * 2 + 50, 70 + height * 0, 200, &g, 0, 255);

        cvui::printf(windowMat, width * 2 + 50, 120 + height * 0, "r");
        cvui::trackbar(windowMat, width * 2 + 50, 130 + height * 0, 200, &r, 0, 255);

        // 低像素差
        cvui::printf(windowMat, width * 1 + 50, 0 + height * 1, "loB");
        cvui::trackbar(windowMat, width * 1 + 50, 10 + height * 1, 200, &loB, 0, 255);

        cvui::printf(windowMat, width * 1 + 50, 60 + height * 1, "loG");
        cvui::trackbar(windowMat, width * 1 + 50, 70 + height * 1, 200, &loG, 0, 255);

        cvui::printf(windowMat, width * 1 + 50, 120 + height * 1, "loR");
        cvui::trackbar(windowMat, width * 1 + 50, 130 + height * 1, 200, &loR, 0, 255);

        // 高像素差
        cvui::printf(windowMat, width * 2 + 50, 0 + height * 1, "upB");
        cvui::trackbar(windowMat, width * 2 + 50, 10 + height * 1, 200, &upB, 0, 255);

        cvui::printf(windowMat, width * 2 + 50, 60 + height * 1, "upG");
        cvui::trackbar(windowMat, width * 2 + 50, 70 + height * 1, 200, &upG, 0, 255);

        cvui::printf(windowMat, width * 2 + 50, 120 + height * 1, "upR");
        cvui::trackbar(windowMat, width * 2 + 50, 130 + height * 1, 200, &upR, 0, 255);

        // 标志
        cvui::printf(windowMat, width * 0 + 50, 60 + height * 2, "flags: default");

        cvui::printf(windowMat, width * 1 + 50, 60 + height * 2, "flags: 4 | FLOODFILL_FIXED_RANGE");

        cvui::printf(windowMat, width * 2 + 50, 60 + height * 2, "flags: 8 | FLOODFILL_FIXED_RANGE");

        // circle
        cv::circle(windowMat, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);

        {
            cv::Rect rect;
            cv::Mat tempMat;
            cv::Mat dstMat;
            // 填充
            tempMat = srcMat.clone();
            cv::floodFill(tempMat,
                          cv::Point(x, y),
                          cv::Scalar(b, g, r),
                          &rect,
                          cv::Scalar(loB, loG, loR),
                          cv::Scalar(upB, upG, upR));
            dstMat = windowMat(cv::Range(srcMat.rows * 3, srcMat.rows * 4),
                               cv::Range(srcMat.cols * 0, srcMat.cols * 1));
            cv::addWeighted(dstMat, 0.0f, tempMat, 1.0f, 0.0f, dstMat);

            // 填充
            tempMat = srcMat.clone();
            cv::floodFill(tempMat,
                          cv::Point(x, y),
                          cv::Scalar(b, g, r),
                          &rect,
                          cv::Scalar(loB, loG, loR),
                          cv::Scalar(upB, upG, upR),
                          4 | cv::FLOODFILL_FIXED_RANGE);
            dstMat = windowMat(cv::Range(srcMat.rows * 3, srcMat.rows * 4),
                               cv::Range(srcMat.cols * 1, srcMat.cols * 2));
            cv::addWeighted(dstMat, 0.0f, tempMat, 1.0f, 0.0f, dstMat);

            // 填充
            tempMat = srcMat.clone();
            cv::floodFill(tempMat,
                          cv::Point(x, y),
                          cv::Scalar(b, g, r),
                          &rect,
                          cv::Scalar(loB, loG, loR),
                          cv::Scalar(upB, upG, upR),
                          8 | cv::FLOODFILL_FIXED_RANGE);
            dstMat = windowMat(cv::Range(srcMat.rows * 3, srcMat.rows * 4),
                               cv::Range(srcMat.cols * 2, srcMat.cols * 3));
            cv::addWeighted(dstMat, 0.0f, tempMat, 1.0f, 0.0f, dstMat);
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
}
