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

#define WINDOW_NAME	"ximgproc"
#define FILEPATH "D:/workspace/sample/lena(1).tiff"

int main(int argc, char *argv[])
{
    QString fileName1 =FILEPATH;
    int width = 400;
    int height = 300;

    cv::Mat srcMat = cv::imread(fileName1.toStdString());
    cv::resize(srcMat, srcMat, cv::Size(width, height));

    cv::String windowName = QString(WINDOW_NAME).toStdString();
    cvui::init(windowName);

    cv::Mat windowMat = cv::Mat(cv::Size(srcMat.cols * 2, srcMat.rows * 3),
                                srcMat.type());

    int threshold1 = 200;
    int threshold2 = 100;
    while(true)
    {
        windowMat = cv::Scalar(0, 0, 0);

        cv::Mat mat;

        cv::Mat tempMat;
        // 原图先copy到左边
        mat = windowMat(cv::Range(srcMat.rows * 0, srcMat.rows * 1),
                        cv::Range(srcMat.cols * 0, srcMat.cols * 1));
        cv::addWeighted(mat, 0.0f, srcMat, 1.0f, 0.0f, mat);

        {
            // 灰度图
            cv::Mat grayMat;
            cv::cvtColor(srcMat, grayMat, cv::COLOR_BGR2GRAY);
            // copy
            mat = windowMat(cv::Range(srcMat.rows * 0, srcMat.rows * 1),
                            cv::Range(srcMat.cols * 1, srcMat.cols * 2));
            cv::Mat grayMat2;
            cv::cvtColor(grayMat, grayMat2, cv::COLOR_GRAY2BGR);
            cv::addWeighted(mat, 0.0f, grayMat2, 1.0f, 0.0f, mat);

            // 均值滤波
            cv::blur(grayMat, tempMat, cv::Size(3, 3));

            cvui::printf(windowMat, width * 1 + 20, height * 1 + 20, "threshold1");
            cvui::trackbar(windowMat, width * 1 + 20, height * 1 + 40, 200, &threshold1, 0, 255);
            cvui::printf(windowMat, width * 1 + 20, height * 1 + 100, "threshold2");
            cvui::trackbar(windowMat, width * 1 + 20, height * 1 + 120, 200, &threshold2, 0, 255);

            // canny边缘检测
            cv::Canny(tempMat, tempMat, threshold1, threshold2);
            // copy
            mat = windowMat(cv::Range(srcMat.rows * 1, srcMat.rows * 2),
                            cv::Range(srcMat.cols * 0, srcMat.cols * 1));
            cv::cvtColor(tempMat, grayMat2, cv::COLOR_GRAY2BGR);
            cv::addWeighted(mat, 0.0f, grayMat2, 1.0f, 0.0f, mat);

            // 查找轮廓
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(tempMat, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

            // 绘制轮廓
            cv::Mat maskers = cv::Mat::zeros(grayMat.size(), CV_32SC1);
            maskers = cv::Scalar::all(0);
            cv::Mat tMat = srcMat.clone();
            tMat = cv::Scalar(0, 0, 0);
            for(int index = 0; index < contours.size(); index++)
            {
                cv::drawContours(maskers, contours, index, cv::Scalar::all(index+1));
                cv::drawContours(tMat, contours, index, cv::Scalar(0, 0, 255));
            }
            // copy
            mat = windowMat(cv::Range(srcMat.rows * 2, srcMat.rows * 3),
                            cv::Range(srcMat.cols * 0, srcMat.cols * 1));
            cv::addWeighted(mat, 0.0f, tMat, 1.0f, 0.0f, mat);

            // 分水岭
            cv::watershed(srcMat, maskers);
            cv::Mat watershedImage(maskers.size(), CV_8UC3) ;
            for(int i = 0 ; i < maskers.rows ; i++ )
            {
                for(int j = 0 ; j < maskers.cols; j++)
                {
                    int index = maskers.at<int>(i, j);
                    if(index == -1)
                    {
                        watershedImage.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
                    }else if( index <= 0 || index > contours.size() )
                    {
                        watershedImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
                    }else
                    {
                        watershedImage.at<cv::Vec3b>(i, j) = cv::Vec3b((index - 5 > 0 ? 0 : index % 5) * 50,
                                                                       (index - 5 > 0 ? index - 5 : 0) % 5 * 50,
                                                                       (index - 10 > 0 ? index - 10 : 0) % 5 * 50);
                    }
                    // 混合灰皮图和 分水岭效果 图 并显 示最终的窗 口
                }
            }
            // copy
            mat = windowMat(cv::Range(srcMat.rows * 2, srcMat.rows * 3),
                            cv::Range(srcMat.cols * 1, srcMat.cols * 2));
            cv::addWeighted(mat, 0.0f, watershedImage, 1.0f, 0.0f, mat);

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
