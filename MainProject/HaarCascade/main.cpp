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

QString WINDOW_NAME	= "Haar Cascades";
QString FOLDERPATH = "D:/workspace/sample/";
QString HAARCASCADES = "D:/Opencv411/etc/haarcascades";

int main(int argc, char *argv[])
{
    cv::String windowName = WINDOW_NAME.toStdString();
    cvui::init(windowName);

    cv::Mat windowMat = cv::Mat(cv::Size(520, 400),
                                CV_8UC3);

    cv::VideoCapture videoCapture;
    videoCapture.open("E:/testFile/4.avi");

    cv::Mat mat;
    cv::Mat dstMat;
    int currentIndex = 0;

    // 级联人脸分类器
    cv::CascadeClassifier cascadeClassifier;
    cascadeClassifier.load((HAARCASCADES+"haarcascade_frontalface_alt.xml").toStdString());
    while(true)
    {
        // 刷新全图黑色
        windowMat = cv::Scalar(0, 0, 0);
        currentIndex = videoCapture.get(cv::CAP_PROP_POS_FRAMES);
        videoCapture >> mat;
        if(!mat.empty())
        {
            dstMat = mat.clone();;

            // 灰度变换
            cv::Mat grayMat;
            cv::cvtColor(dstMat, grayMat, COLOR_BGR2GRAY);
            // 直方图均衡化
            cv::Mat histMat;
            cv::equalizeHist(grayMat, histMat);
            // 多尺度人脸检测
            std::vector<cv::Rect> faces;
            cascadeClassifier.detectMultiScale(histMat,
                                               faces,
                                               1.1,
                                               3,
                                               0 | cv::CASCADE_SCALE_IMAGE,
                                               cv::Size(10, 10));
            qDebug() << __FILE__ << __LINE__ << "faces number:" << faces.size();
            // 人脸检测结果判定
            for(int index = 0; index < faces.size(); index++)
            {
                // 检测到人脸
                cv::rectangle(dstMat, faces.at(index), cv::Scalar(0, 0, 255));
            }

            // 视频复制
            mat = windowMat(cv::Range(0, 0 + dstMat.rows),
                            cv::Range(0, 0 + dstMat.cols));
            cv::addWeighted(mat, 0.0f, dstMat, 1.0f, 0.0f, mat);
        }
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
