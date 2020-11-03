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

QString WINDOW_NAME	= "Find Contours";
QString FOLDERPATH = "D:/workspace/sample/";

int main(int argc, char *argv[])
{
    QString fileName1 = "E:/testFile/templ2.jpg";

    cv::Mat srcMat = cv::imread(fileName1.toStdString());

    cv::String windowName = WINDOW_NAME.toStdString();
    cvui::init(windowName);

    cv::Mat windowMat = cv::Mat(cv::Size(700, 900),
                                srcMat.type());

    cv::VideoCapture videoCapture;
    videoCapture.open("E:/testFile/test.avi");

    cv::Mat mat;
    cv::Mat dstMat;
    int currentIndex = 0;
    while(true)
    {
        // 刷新全图黑色
        windowMat = cv::Scalar(0, 0, 0);
        currentIndex = videoCapture.get(cv::CAP_PROP_POS_FRAMES);
        videoCapture >> mat;
        if(!mat.empty())
        {
            dstMat = mat.clone();

            // 模板头像
            mat = windowMat(cv::Range(srcMat.rows * 0, srcMat.rows * 1),
                            cv::Range(srcMat.cols * 0, srcMat.cols * 1));
            cv::addWeighted(mat, 0.0f, srcMat, 1.0f, 0.0f, mat);

            // 模板
            cv::Mat result;

            // 平方差匹配法
            {
                cv::matchTemplate(dstMat, srcMat, result, cv::TM_SQDIFF);
                // 获取结果中的最大值、最小值的下标和对应值
                double minVal, maxVal;
                cv::Point minIdx, maxIdx;
                cv::minMaxLoc(result, &minVal, &maxVal, &minIdx, &maxIdx);
                cv::rectangle(dstMat,
                              cv::Rect(minIdx.x,
                                       minIdx.y,
                                       srcMat.cols,
                                       srcMat.rows),
                              cv::Scalar(0, 0, 255));
                cvui::printf(windowMat,
                             srcMat.cols * 0,
                             srcMat.rows * 1 + dstMat.rows + 30 * 0,
                             0.5f,
                             0xFF0000,
                             "TM_SQDIFF minVal = %f",
                             minVal);
            }
            // 归一平方差匹配法
            {
                cv::matchTemplate(dstMat, srcMat, result, cv::TM_SQDIFF_NORMED);
                // 获取结果中的最大值、最小值的下标和对应值
                double minVal, maxVal;
                cv::Point minIdx, maxIdx;
                cv::minMaxLoc(result, &minVal, &maxVal, &minIdx, &maxIdx);
                cv::rectangle(dstMat,
                              cv::Rect(minIdx.x,
                                       minIdx.y,
                                       srcMat.cols,
                                       srcMat.rows),
                              cv::Scalar(0, 255, 0));
                cvui::printf(windowMat,
                             srcMat.cols * 0,
                             srcMat.rows * 1 + dstMat.rows + 30 * 1,
                             0.5f,
                             0x00FF00,
                             "TM_SQDIFF_NORMED minVal = %f",
                             minVal);
            }
            // 相关匹配法
            {
                cv::matchTemplate(dstMat, srcMat, result, cv::TM_CCORR);
                // 获取结果中的最大值、最小值的下标和对应值
                double minVal, maxVal;
                cv::Point minIdx, maxIdx;
                cv::minMaxLoc(result, &minVal, &maxVal, &minIdx, &maxIdx);
                cv::rectangle(dstMat,
                              cv::Rect(maxIdx.x,
                                       maxIdx.y,
                                       srcMat.cols,
                                       srcMat.rows),
                              cv::Scalar(255, 0, 0));
                cvui::printf(windowMat,
                             srcMat.cols * 0,
                             srcMat.rows * 1 + dstMat.rows + 30 * 2,
                             0.5f,
                             0x0000FF,
                             "TM_CCORR maxVal = %f",
                             maxVal);
            }
            // 归一相关匹配法
            {
                cv::matchTemplate(dstMat, srcMat, result, cv::TM_CCORR_NORMED);
                // 获取结果中的最大值、最小值的下标和对应值
                double minVal, maxVal;
                cv::Point minIdx, maxIdx;
                cv::minMaxLoc(result, &minVal, &maxVal, &minIdx, &maxIdx);
                cv::rectangle(dstMat,
                              cv::Rect(maxIdx.x,
                                       maxIdx.y,
                                       srcMat.cols,
                                       srcMat.rows),
                              cv::Scalar(255, 255, 0));
                cvui::printf(windowMat,
                             srcMat.cols * 0,
                             srcMat.rows * 1 + dstMat.rows + 30 * 3,
                             0.5f,
                             0x00FFFF,
                             "TM_CCORR maxVal = %f",
                             maxVal);
            }
            // 系数匹配法
            {
                cv::matchTemplate(dstMat, srcMat, result, cv::TM_CCOEFF);
                // 获取结果中的最大值、最小值的下标和对应值
                double minVal, maxVal;
                cv::Point minIdx, maxIdx;
                cv::minMaxLoc(result, &minVal, &maxVal, &minIdx, &maxIdx);
                cv::rectangle(dstMat,
                              cv::Rect(maxIdx.x,
                                       maxIdx.y,
                                       srcMat.cols,
                                       srcMat.rows),
                              cv::Scalar(255, 0, 255));
                cvui::printf(windowMat,
                             srcMat.cols * 0,
                             srcMat.rows * 1 + dstMat.rows + 30 * 4,
                             0.5f,
                             0xFF00FF,
                             "TM_CCOEFF maxVal = %f",
                             maxVal);
            }
            // 系数匹配法匹配法
            {
                cv::matchTemplate(dstMat, srcMat, result, cv::TM_CCOEFF_NORMED);
                // 获取结果中的最大值、最小值的下标和对应值
                double minVal, maxVal;
                cv::Point minIdx, maxIdx;
                cv::minMaxLoc(result, &minVal, &maxVal, &minIdx, &maxIdx);
                cv::rectangle(dstMat,
                              cv::Rect(maxIdx.x,
                                       maxIdx.y,
                                       srcMat.cols,
                                       srcMat.rows),
                              cv::Scalar(255, 255, 255));
                cvui::printf(windowMat,
                             srcMat.cols * 0,
                             srcMat.rows * 1 + dstMat.rows + 30 * 5,
                             0.5f,
                             0xFFFFFF,
                             "TM_CCOEFF_NORMED maxVal = %f",
                             maxVal);
            }

            // 视频复制
            mat = windowMat(cv::Range(srcMat.rows * 1, srcMat.rows * 1 + dstMat.rows),
                            cv::Range(srcMat.cols * 0, srcMat.cols * 0 + dstMat.cols));
            cv::addWeighted(mat, 0.0f, dstMat, 1.0f, 0.0f, mat);

        }

        qDebug() << __FILE__ << __LINE__ << currentIndex;
        // 更新
        cvui::update();
        // 显示
        cv::imshow(windowName, windowMat);
        // esc键退出
        int key = cv::waitKey(0);
        switch (key) {
        case 97:        // 'a' 往前一帧
            currentIndex--;
            if(currentIndex < 0)
            {
                currentIndex = 0;
            }
            videoCapture.set(cv::CAP_PROP_POS_FRAMES, currentIndex);
            break;
        case 115:       // ‘s’ 往后一帧
            break;
        default:
            break;
        }
        if(key == 27)
        {
            break;
        }
    }
}
