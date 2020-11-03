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

void testCvuiRunNormal()
{
    cv::String windowName = "testCvuiRunNormal";
    cvui::init(windowName);
    //                      高度  宽度
    cv::Mat frame = cv::Mat(300, 400, CV_8UC3);
    int count = 0;
    while(true)
    {
        frame = cv::Scalar(100, 100, 100);
        if(cvui::button(frame, 100, 100, "testButton"))
        {
            count++;
        }
        cvui::printf(frame, 100, 200, 1.0, 0xFF0000, "count = %d", count);
        cvui::update();
        cv::imshow(windowName, frame);

        // Check if ESC key was pressed
        if (cv::waitKey(20) == 27)
        {
            break;
        }
    }
}

void testBaseDraw()
{
    cv::Mat mat(400, 400, CV_8UC3, cv::Scalar());
    while(true)
    {
        mat = cv::Scalar();
        cv::imshow("1", mat);
        cv::waitKey(0);
        cv::putText(mat, "Hello world!!!", cv::Point(0, 200), cv::FONT_HERSHEY_COMPLEX,
                    1, cv::Scalar(0, 0, 255));
        cv::imshow("1", mat);
        cv::waitKey(0);
        // line
        cv::line(mat, cv::Point(30, 30) , cv::Point(370, 30) , cv::Scalar(255, 255, 255), 1);
        cv::line(mat, cv::Point(370, 30), cv::Point(370, 370), cv::Scalar(255, 255, 255), 1);
        cv::line(mat, cv::Point(370, 370), cv::Point(30, 370), cv::Scalar(255, 255, 255), 1);
        cv::line(mat, cv::Point(30, 370), cv::Point(30 , 30), cv::Scalar(255, 255, 255), 1);
        cv::imshow("1", mat);
        cv::waitKey(0);
        // ellipse
        cv::ellipse(mat, cv::Point(50+1, 50+1) , cv::Size(10, 20), 45.0, 0.0, 360.0,
                    cv::Scalar(0, 0, 255), 1);
        cv::ellipse(mat, cv::Point(50+1, 350-1), cv::Size(10, 20), 135.0, 0.0, 360.0,
                    cv::Scalar(0, 0, 255), -1);
        cv::ellipse(mat, cv::Point(350-1, 50+1), cv::Size(20, 10), 45.0, 0.0, 360.0,
                    cv::Scalar(0, 0, 255), 1);
        cv::ellipse(mat, cv::Point(350-1, 350-1), cv::Size(20, 10), 135.0, 0.0, 360.0,
                    cv::Scalar(0, 0, 255), -1);
        cv::imshow("1", mat);
        cv::waitKey(0);
        // rectangle
        cv::rectangle(mat, cv::Rect(100   , 100   , 20, 20), cv::Scalar(0, 255, 0));
        cv::rectangle(mat, cv::Rect(100   , 300-20, 20, 20), cv::Scalar(0, 255, 0));
        cv::rectangle(mat, cv::Rect(300-20, 100   , 20, 20), cv::Scalar(0, 255, 0), -1);
        cv::rectangle(mat, cv::Rect(300-20, 300-20, 20, 20), cv::Scalar(0, 255, 0), -1);
        cv::imshow("1", mat);
        cv::waitKey(0);
        // circle
        cv::circle(mat, cv::Point(200, 200), 10, cv::Scalar(200, 200, 200), -1);
        cv::circle(mat, cv::Point(200, 200), 20, cv::Scalar(200, 200, 200), 1);
        cv::circle(mat, cv::Point(200, 200), 30, cv::Scalar(200, 200, 200), 2);
        cv::circle(mat, cv::Point(200, 200), 40, cv::Scalar(200, 200, 200), 3);
        cv::circle(mat, cv::Point(200, 200), 50, cv::Scalar(200, 200, 200), 4);
        cv::imshow("1", mat);
        cv::waitKey(0);
        // fillPoly
        cv::Point rootPoints[1][4];
        rootPoints[0][0] = cv::Point(200, 150);
        rootPoints[0][1] = cv::Point(250, 200);
        rootPoints[0][2] = cv::Point(200, 250);
        rootPoints[0][3] = cv::Point(150, 200);
        const cv::Point * ppt[1] = { rootPoints[0] };
        const int npt[] = {4};
        cv::fillPoly(mat, ppt, npt, 1, cv::Scalar(0, 255, 255));
        cv::imshow("1", mat);
        cv::waitKey(0);
    }
}

void testCommonOperate()
{
#define TEST_GET_TICK_COUNT (1)
#define TEST_ROTATE_90      (1)

#if TEST_GET_TICK_COUNT     // 测试计时函数
    for(int index = 0; index < 10; index++)
    {
        int64 tickCount = cv::getTickCount();
        qDebug() << __FUNCTION__ << __LINE__ << "===================== test cv::getTickCount(), now times:" << index;
        qDebug() << __FUNCTION__ << __LINE__ << "cv::getTickCount() =" << tickCount;
        // Qt的线程睡眠函数
        QThread::msleep(200 * index);
        int64 tickCount2 = cv::getTickCount();
        qDebug() << __FUNCTION__ << __LINE__ << "cv::getTickCount() =" << tickCount2;
        int64 ms = ((double)tickCount2 - tickCount) * 1000.0f/ cv::getTickFrequency();
        qDebug() << __FUNCTION__ << __LINE__ << "cv::getTickFrequency() =" << (int64)cv::getTickFrequency() << ", ms:" << ms;
    }
#endif

#if TEST_ROTATE_90
    cv::Mat srcMat;
    QString fileName = "E:/workspace/sample/lena.tiff";
    srcMat = cv::imread(fileName.toStdString());
    if(!srcMat.data)
    {
        qDebug() << __FILE__ << __LINE__ << "Failed to load image:" << fileName;
        return;
    }
    cv::imshow("OpenCVDemo v1.5.0 QQ:21497936 blog:blog.csdn.net/qq21497936", srcMat);
    float scaleStep = 0.05f;
    while(true)
    {
        int key = cv::waitKey();
        if(key == '1')          // 逆时钟旋转90度
        {
            int64 tickCount = cv::getTickCount();
            qDebug() << __FUNCTION__ << __LINE__ << "cv::getTickCount() =" << tickCount;
            cv::rotate(srcMat, srcMat, cv::ROTATE_90_COUNTERCLOCKWISE);
            int64 tickCount2 = cv::getTickCount();
            qDebug() << __FUNCTION__ << __LINE__ << "cv::getTickCount() =" << tickCount2;
            int64 ms = ((double)tickCount2 - tickCount) * 1000.0f/ cv::getTickFrequency();
            qDebug() << __FUNCTION__ << __LINE__ << "take time ms:" << ms;
            cv::imshow("OpenCVDemo v1.5.0 QQ:21497936 blog:blog.csdn.net/qq21497936", srcMat);
        }else if(key == '2')    // 顺时钟旋转90度
        {
            int64 tickCount = cv::getTickCount();
            qDebug() << __FUNCTION__ << __LINE__ << "cv::getTickCount() =" << tickCount;
            cv::rotate(srcMat, srcMat, cv::ROTATE_90_CLOCKWISE);
            int64 tickCount2 = cv::getTickCount();
            qDebug() << __FUNCTION__ << __LINE__ << "cv::getTickCount() =" << tickCount2;
            int64 ms = ((double)tickCount2 - tickCount) * 1000.0f/ cv::getTickFrequency();
            qDebug() << __FUNCTION__ << __LINE__ << "take time ms:" << ms;
            cv::imshow("OpenCVDemo v1.5.0 QQ:21497936 blog:blog.csdn.net/qq21497936", srcMat);
        }else if(key == '3')    // x轴翻转（镜像）
        {
            int64 tickCount = cv::getTickCount();
            qDebug() << __FUNCTION__ << __LINE__ << "cv::getTickCount() =" << tickCount;
            cv::flip(srcMat, srcMat, 0);
            int64 tickCount2 = cv::getTickCount();
            qDebug() << __FUNCTION__ << __LINE__ << "cv::getTickCount() =" << tickCount2;
            int64 ms = ((double)tickCount2 - tickCount) * 1000.0f/ cv::getTickFrequency();
            qDebug() << __FUNCTION__ << __LINE__ << "take time ms:" << ms;
            cv::imshow("OpenCVDemo v1.5.0 QQ:21497936 blog:blog.csdn.net/qq21497936", srcMat);
        }else if(key == '4')    // y轴翻转（镜像）
        {
            int64 tickCount = cv::getTickCount();
            qDebug() << __FUNCTION__ << __LINE__ << "cv::getTickCount() =" << tickCount;
            cv::flip(srcMat, srcMat, 1);
            int64 tickCount2 = cv::getTickCount();
            qDebug() << __FUNCTION__ << __LINE__ << "cv::getTickCount() =" << tickCount2;
            int64 ms = ((double)tickCount2 - tickCount) * 1000.0f/ cv::getTickFrequency();
            qDebug() << __FUNCTION__ << __LINE__ << "take time ms:" << ms;
            cv::imshow("OpenCVDemo v1.5.0 QQ:21497936 blog:blog.csdn.net/qq21497936", srcMat);
        }else if(key == '5')    // 缩小
        {
            int64 tickCount = cv::getTickCount();
            qDebug() << __FUNCTION__ << __LINE__ << "cv::getTickCount() =" << tickCount;
            cv::resize(srcMat, srcMat, cv::Size((int)(srcMat.cols * (1.0f - scaleStep)),
                                                (int)(srcMat.rows * (1.0f - scaleStep))));
            int64 tickCount2 = cv::getTickCount();
            qDebug() << __FUNCTION__ << __LINE__ << "cv::getTickCount() =" << tickCount2;
            int64 ms = ((double)tickCount2 - tickCount) * 1000.0f/ cv::getTickFrequency();
            qDebug() << __FUNCTION__ << __LINE__ << "take time ms:" << ms;
            cv::imshow("OpenCVDemo v1.5.0 QQ:21497936 blog:blog.csdn.net/qq21497936", srcMat);
        }else if(key == '6')    // 放大
        {
            int64 tickCount = cv::getTickCount();
            qDebug() << __FUNCTION__ << __LINE__ << "cv::getTickCount() =" << tickCount;
            cv::resize(srcMat, srcMat, cv::Size((int)(srcMat.cols * (1.0f + scaleStep)),
                                                (int)(srcMat.rows * (1.0f + scaleStep))));
            int64 tickCount2 = cv::getTickCount();
            qDebug() << __FUNCTION__ << __LINE__ << "cv::getTickCount() =" << tickCount2;
            int64 ms = ((double)tickCount2 - tickCount) * 1000.0f/ cv::getTickFrequency();
            qDebug() << __FUNCTION__ << __LINE__ << "take time ms:" << ms;
            cv::imshow("OpenCVDemo v1.5.0 QQ:21497936 blog:blog.csdn.net/qq21497936", srcMat);
        }
        if(key == 27)
        {
            break;
        }
    }
#endif
}

void testAffineMap()
{
    QString fileName1 ="E:/workspace/sample/lena.tiff";
    cv::Mat srcMat = cv::imread(fileName1.toStdString());
    cv::Mat dstMat;
    int width = 400;
    int height = 300;

    cv::resize(srcMat, srcMat, cv::Size(width, height));

    cv::Mat windowMat = cv::Mat(cv::Size(srcMat.cols * 2,
                                         srcMat.rows * 4),
                                srcMat.type());

    while(true)
    {
        windowMat = cv::Scalar(0, 0, 0);

        cv::Mat mat = windowMat(cv::Range(srcMat.rows * 0, srcMat.rows * 1),
                                cv::Range(srcMat.cols * 0, srcMat.cols * 1));
        cv::addWeighted(mat, 0.0f, srcMat, 1.0f, 0.0f, mat);

        // 第一种旋转180度
        {
            cv::Mat M = cv::getRotationMatrix2D(cv::Point(srcMat.cols / 2,
                                                          srcMat.rows / 2),
                                                180.0f,
                                                1.0f);
            dstMat = srcMat.clone();
            dstMat = cv::Scalar(0, 0, 0);
            cv::warpAffine(srcMat, dstMat, M, cv::Size(srcMat.cols, srcMat.rows));
            mat = windowMat(cv::Range(srcMat.rows * 0, srcMat.rows * 1),
                            cv::Range(srcMat.cols * 1, srcMat.cols * 2));
            cv::addWeighted(mat, 0.0f, dstMat, 1.0f, 0.0f, mat);
        }

        // 第二种旋转45度,缩小1/2
        {
            cv::Mat M = cv::getRotationMatrix2D(cv::Point(srcMat.cols / 2,
                                                          srcMat.rows / 2),
                                                45.0f,
                                                0.5f);
            dstMat = srcMat.clone();
            dstMat = cv::Scalar(0, 0, 0);
            cv::warpAffine(srcMat, dstMat, M, cv::Size(srcMat.cols, srcMat.rows));
            mat = windowMat(cv::Range(srcMat.rows * 1, srcMat.rows * 2),
                            cv::Range(srcMat.cols * 0, srcMat.cols * 1));
            cv::addWeighted(mat, 0.0f, dstMat, 1.0f, 0.0f, mat);
        }
        // 第三种旋转315度，缩小1/2
        {
            cv::Mat M = cv::getRotationMatrix2D(cv::Point(srcMat.cols / 2,
                                                          srcMat.rows / 2),
                                                315.0f,
                                                0.5f);
            dstMat = srcMat.clone();
            dstMat = cv::Scalar(0, 0, 0);
            cv::warpAffine(srcMat, dstMat, M, cv::Size(srcMat.cols, srcMat.rows));
            mat = windowMat(cv::Range(srcMat.rows * 1, srcMat.rows * 2),
                            cv::Range(srcMat.cols * 1, srcMat.cols * 2));
            cv::addWeighted(mat, 0.0f, dstMat, 1.0f, 0.0f, mat);
        }
        // 第四种旋转135度，缩小1/2
        {
            cv::Mat M = cv::getRotationMatrix2D(cv::Point(srcMat.cols / 2,
                                                          srcMat.rows / 2),
                                                135.0f,
                                                0.5f);
            dstMat = srcMat.clone();
            dstMat = cv::Scalar(0, 0, 0);
            cv::warpAffine(srcMat, dstMat, M, cv::Size(srcMat.cols, srcMat.rows));
            mat = windowMat(cv::Range(srcMat.rows * 2, srcMat.rows * 3),
                            cv::Range(srcMat.cols * 0, srcMat.cols * 1));
            cv::addWeighted(mat, 0.0f, dstMat, 1.0f, 0.0f, mat);
        }
        // 第五种旋转225度，缩小1/2
        {
            cv::Mat M = cv::getRotationMatrix2D(cv::Point(srcMat.cols / 2,
                                                          srcMat.rows / 2),
                                                225.0f,
                                                0.5f);
            dstMat = srcMat.clone();
            dstMat = cv::Scalar(0, 0, 0);
            cv::warpAffine(srcMat, dstMat, M, cv::Size(srcMat.cols, srcMat.rows));
            mat = windowMat(cv::Range(srcMat.rows * 2, srcMat.rows * 3),
                            cv::Range(srcMat.cols * 1, srcMat.cols * 2));
            cv::addWeighted(mat, 0.0f, dstMat, 1.0f, 0.0f, mat);
        }
        // 第六种使用三角点进行仿射变换，沿着对角线翻转
        {
            cv::Point2f srcTraingle[3];
            cv::Point2f dstTraingle[3];
            srcTraingle[0] = cv::Point2f(0, 0);
            srcTraingle[1] = cv::Point2f(srcMat.cols - 1, 0);
            srcTraingle[2] = cv::Point2f(0, srcMat.rows - 1);
            dstTraingle[0] = cv::Point2f(0, 0);
            dstTraingle[1] = cv::Point2f(0, srcMat.rows - 1);
            dstTraingle[2] = cv::Point2f(srcMat.cols - 1, 0);

            cv::Mat M = cv::getAffineTransform(srcTraingle, dstTraingle);
            dstMat = srcMat.clone();
            dstMat = cv::Scalar(0, 0, 0);
            cv::warpAffine(srcMat, dstMat, M, cv::Size(srcMat.cols, srcMat.rows));
            mat = windowMat(cv::Range(srcMat.rows * 3, srcMat.rows * 4),
                            cv::Range(srcMat.cols * 0, srcMat.cols * 1));
            cv::addWeighted(mat, 0.0f, dstMat, 1.0f, 0.0f, mat);
        }
        // 第七种使用三角点进行仿射变换
        {
            cv::Point2f srcTraingle[3];
            cv::Point2f dstTraingle[3];
            srcTraingle[0] = cv::Point2f(0, 0);
            srcTraingle[1] = cv::Point2f(srcMat.cols - 1, 0);
            srcTraingle[2] = cv::Point2f(0, srcMat.rows - 1);
            dstTraingle[0] = cv::Point2f(srcMat.cols / 4, srcMat.rows / 4);
            dstTraingle[1] = cv::Point2f(srcMat.cols / 4 * 3, srcMat.rows / 4 );
            dstTraingle[2] = cv::Point2f(srcMat.cols / 2, srcMat.rows - 1);

            cv::Mat M = cv::getAffineTransform(srcTraingle, dstTraingle);
            dstMat = srcMat.clone();
            dstMat = cv::Scalar(0, 0, 0);
            cv::warpAffine(srcMat, dstMat, M, cv::Size(srcMat.cols, srcMat.rows));
            mat = windowMat(cv::Range(srcMat.rows * 3, srcMat.rows * 4),
                            cv::Range(srcMat.cols * 1, srcMat.cols * 2));
            cv::addWeighted(mat, 0.0f, dstMat, 1.0f, 0.0f, mat);
        }

        // 显示
        cv::imshow("windowName", windowMat);
        // esc键退出
        if(cv::waitKey(25) == 27)
        {
            break;
        }
    }
}

void testContrastAndBrightness()
{
    cv::Mat mat1;
    QString fileName1 = "E:/workspace/sample/lena.tiff";
    mat1 = cv::imread(fileName1.toStdString());

    cv::String windowName = "openCVDemo v1.8.0";
    cvui::init(windowName);

    if(!mat1.data)
    {
        qDebug() << __FILE__ << __LINE__
                 << "Failed to load image:" << fileName1;
        return;
    }
    // 增强对比度
    float r;
    float g;
    float b;
    cv::Mat dstMat;
    dstMat = cv::Mat::zeros(mat1.size(), mat1.type());
    cv::Mat windowMat = cv::Mat(cv::Size(dstMat.cols * 2, dstMat.rows), CV_8UC3);
    int alpha = 100;    // 小于1，则降低对比度
    int beta = 0;     // 负数，则降低亮度
    cvui::window(windowMat, dstMat.cols, 0, dstMat.cols, dstMat.rows, "settings");
    while(true)
    {
        windowMat = cv::Scalar(0, 0, 0);
        cvui::printf(windowMat, 375, 40, "contrast");
        cvui::trackbar(windowMat, 375, 50, 165, &alpha, 0, 400);
        cvui::printf(windowMat, 375, 100, "brightness");
        cvui::trackbar(windowMat, 375, 110, 165, &beta, -255, 255);
#if 1
        for(int row = 0; row < mat1.rows; row++)
        {
            for(int col = 0; col < mat1.cols; col++)
            {
                b = mat1.at<cv::Vec3b>(row, col)[0];
                g = mat1.at<cv::Vec3b>(row, col)[1];
                r = mat1.at<cv::Vec3b>(row, col)[2];
#if 0
                // 改变背景色
                if(r > 200 && g > 200 && b > 200)
                {
                    r = 25/4.0;
                    g = 246.0f;
                    b = 197.0f;
                    dstMat.at<cv::Vec3b>(row, col)[0] = b;
                    dstMat.at<cv::Vec3b>(row, col)[1] = g;
                    dstMat.at<cv::Vec3b>(row, col)[2] = r;
                }
#endif
#if 1
                // 对比度、亮度计算公式 cv::saturate_cast<uchar>(value)：防止溢出
                dstMat.at<cv::Vec3b>(row, col)[0] = cv::saturate_cast<uchar>(b * alpha / 100.0f + beta);
                dstMat.at<cv::Vec3b>(row, col)[1] = cv::saturate_cast<uchar>(g * alpha / 100.0f + beta);
                dstMat.at<cv::Vec3b>(row, col)[2] = cv::saturate_cast<uchar>(r * alpha / 100.0f + beta);
#else
                // 对比度、亮度计算公式 cv::saturate_cast<uchar>(value)：防止溢出
                dstMat.at<cv::Vec3b>(row, col)[0] = cv::saturate_cast<uchar>(b * alpha / 100.0f + beta);
                dstMat.at<cv::Vec3b>(row, col)[1] = cv::saturate_cast<uchar>(g * alpha / 100.0f + beta);
                dstMat.at<cv::Vec3b>(row, col)[2] = cv::saturate_cast<uchar>(r * alpha / 100.0f + beta);
#endif
            }
        }
//        cvui::trackbar(windowMat, dstMat.cols, dstMat.rows / 3, dstMat.cols, &?alpha, 0, 100, );
        cv::Mat imageMat = windowMat(cv::Range(0, dstMat.rows), cv::Range(0, dstMat.cols));
        cv::addWeighted(imageMat, 0.0, dstMat, 1.0, 0.0f, imageMat);
#endif
        cvui::update();
        cv::imshow(windowName, windowMat);
        if(cv::waitKey(25) == 27)
        {
            break;
        }
    }
}

void testROIAndBlend()
{
#define TEST_ROI        (0)     // roi
#define TEST_BLEND      (0)     // 图像加权混合（全部区域）
#define TEST_BLEND_ROI  (1)     // 图像加权混合（roi区域）

#if TEST_ROI
    // 测试提取
    cv::Mat srcMat;
    QString fileName = "D:/qtProject/openCVDemo/openCVDemo/modules/openCVManager/images/1.jpg";
    srcMat = cv::imread(fileName.toStdString());
    if(!srcMat.data)
    {
        qDebug() << __FILE__ << __LINE__ << "Failed to load image:" << fileName;
        return;
    }
    cv::imshow("orgin mat", srcMat);
    cv::Mat roiMat = srcMat(cv::Range(srcMat.rows/2 - 50, srcMat.rows/2 + 50),
                            cv::Range(srcMat.cols/2 - 30, srcMat.cols/2 + 30));
    cv::imshow("roi mat", roiMat);
    cv::waitKey(0);
#endif

#if TEST_BLEND
    // 测试线性混合: 正向 与 x轴翻转后的线性混合
    cv::Mat mat1;
    cv::Mat mat2;
    cv::Mat mat3;
    QString fileName1 = "D:/qtProject/openCVDemo/openCVDemo/modules/openCVManager/images/1.jpg";
    QString fileName2 = "D:/qtProject/openCVDemo/openCVDemo/modules/openCVManager/images/1.bmp";
    mat1 = cv::imread(fileName1.toStdString());
    mat2 = cv::imread(fileName2.toStdString());
    if(!mat1.data || !mat2.data)
    {
        qDebug() << __FILE__ << __LINE__
                 << "Failed to load image:" << fileName1 << "or" << fileName2;
        return;
    }
    // 对mat2进行缩放，缩放至mat1的大小,否则会报错
    cv::resize(mat2, mat2, cv::Size(mat1.cols, mat1.rows));
    // 然后进行混合
    double a = 0.0;
    while(true)
    {
        cv::addWeighted(mat1, a, mat2, (1.0-a), 0.0, mat3);
        cv::imshow("mat3", mat3);
        int key = cv::waitKey(0);
        if(key == 27)
        {
            break;
        }else if(key == '1')
        {
            a += 0.05;
            if(a >= 1.0)
            {
                a = 1.0;
            }
        }else if(key == '2')
        {
            a -= 0.05;
            if(a < 0.0)
            {
                a = 0.0;
            }
        }
    }
#endif

#if TEST_BLEND_ROI
    // 测试线性混合: 正向 与 x轴翻转后的线性混合
    cv::Mat mat1;
    cv::Mat mat2;
    cv::Mat mat3;
    QString fileName1 = "E:/workspace/sample/lena.tiff";
    QString fileName2 = "E:/workspace/sample/man.tiff";
    mat1 = cv::imread(fileName1.toStdString());
    mat2 = cv::imread(fileName2.toStdString());
    if(!mat1.data || !mat2.data)
    {
        qDebug() << __FILE__ << __LINE__
                 << "Failed to load image:" << fileName1 << "or" << fileName2;
        return;
    }
    // 对mat2进行缩放，缩放至mat1的1/4的大小
    cv::resize(mat2, mat2, cv::Size(mat1.cols/2, mat1.rows/2));
    double a = 0.0;
    while(true)
    {
        // mat4只是mat1的副本
    //        cv::Mat mat4 = mat1;
        cv::Mat mat4 = mat1.clone();
        qDebug() << __FILE__ << __LINE__
                 << mat1.cols/2 - mat1.cols/4
                 << mat1.cols/2 + mat1.cols/4
                 << mat1.rows/2 - mat1.rows/4
                 << mat1.rows/2 + mat1.rows/4;
        cv::Mat mat5 = mat4(cv::Range(mat1.rows/2 - mat1.rows/4, mat1.rows/2 + mat1.rows/4),
                            cv::Range(mat1.cols/2 - mat1.cols/4, mat1.cols/2 + mat1.cols/4));
        qDebug() <<__FILE__<<__LINE__;
        cv::addWeighted(mat5, a, mat2, (1.0-a), 0.0, mat5);
        qDebug() <<__FILE__<<__LINE__;
        cv::imshow("mat4", mat4);
        int key = cv::waitKey(0);
        if(key == 27)
        {
            break;
        }else if(key == '1')
        {
            a += 0.05;
            if(a >= 1.0)
            {
                a = 1.0;
            }
        }else if(key == '2')
        {
            a -= 0.05;
            if(a < 0.0)
            {
                a = 0.0;
            }
        }
    }
#endif
}

void testSplitAndMerge()
{
#define TEST_SPLIT          (0)
#define TEST_MERGE          (0)
#define TEST_SPLIT_MERGE    (1)

#if TEST_SPLIT
    // 测试通道分离
    cv::Mat mat1;
    QString fileName1 = "D:/qtProject/openCVDemo/openCVDemo/modules/openCVManager/images/1.jpg";
    mat1 = cv::imread(fileName1.toStdString());
    if(!mat1.data)
    {
        qDebug() << __FILE__ << __LINE__
                 << "Failed to load image:" << fileName1;
        return;
    }
    std::vector<cv::Mat> listChannel;
    cv::split(mat1,listChannel);
    cv::imshow("origin", mat1);
    for(int index = 0; index < listChannel.size(); index++)
    {
        cv::imshow("channel" + std::to_string(index), listChannel.at(index));
    }
    cv::waitKey(0);
#endif


#if TEST_MERGE
    // 测试通道分离
    cv::Mat matR(300, 300, CV_8UC1);
    cv::Mat matG(300, 300, CV_8UC1);
    cv::Mat matB(300, 300, CV_8UC1);
    // 上部分1/3为红色, 2/3为绿色，3/3为蓝色
    for(int row = 0; row < matR.rows; row++)
    {
        if(row < matR.rows / 3)
        {
            for(int col = 0; col < matR.cols; col++)
            {
                matR.at<uchar>(row, col) = 255;
                matG.at<uchar>(row, col) = 0;
                matB.at<uchar>(row, col) = 0;
            }
        }else if(row < matR.rows / 3 * 2 )
        {
            for(int col = 0; col < matR.cols; col++)
            {
                matR.at<uchar>(row, col) = 0;
                matG.at<uchar>(row, col) = 255;
                matB.at<uchar>(row, col) = 0;
            }
        }else{
            for(int col = 0; col < matR.cols; col++)
            {
                matR.at<uchar>(row, col) = 0;
                matG.at<uchar>(row, col) = 0;
                matB.at<uchar>(row, col) = 255;
            }
        }
    }
    cv::Mat matOut;
    std::vector<cv::Mat> listChanel;
    // 特别注意，OpenCV通道顺序为BGR
    listChanel.push_back(matB);
    listChanel.push_back(matG);
    listChanel.push_back(matR);
    cv::merge(listChanel, matOut);
    cv::imshow("merge", matOut);
    cv::waitKey(0);
#endif

#if TEST_SPLIT_MERGE
    // 测试通道分离
    cv::Mat mat1;
    QString fileName1 = "E:/workspace/sample/lena(1).tiff";
    mat1 = cv::imread(fileName1.toStdString());
    if(!mat1.data)
    {
        qDebug() << __FILE__ << __LINE__
                 << "Failed to load image:" << fileName1;
        return;
    }
    std::vector<cv::Mat> listChannel;
    cv::split(mat1,listChannel);
    cv::imshow("origin", mat1);
    // 生成一个空矩阵
    cv::Mat chanel(mat1.rows, mat1.cols, CV_8UC1);
    for(int row = 0; row < chanel.rows; row++)
    {
        for(int col = 0; col < chanel.cols; col++)
        {
            mat1.at<uchar>(row, col) = 0;
        }
    }
    for(int index = 0; index < listChannel.size(); index++)
    {
        // 通道BGR
        if(index == 0)
        {
            std::vector<cv::Mat> listOutChannel;
            listOutChannel.push_back(listChannel.at(0));
            listOutChannel.push_back(chanel);
            listOutChannel.push_back(chanel);
            cv::Mat matOut;

            cv::merge(listOutChannel, matOut);
            cv::imshow("channel" + std::to_string(index), matOut);

        }else if(index == 1)
        {
            std::vector<cv::Mat> listOutChannel;
            listOutChannel.push_back(chanel);
            listOutChannel.push_back(listChannel.at(1));
            listOutChannel.push_back(chanel);
            cv::Mat matOut;
            cv::merge(listOutChannel, matOut);
            cv::imshow("channel" + std::to_string(index), matOut);

        }else if(index == 2)
        {
            std::vector<cv::Mat> listOutChannel;
            listOutChannel.push_back(chanel);
            listOutChannel.push_back(chanel);
            listOutChannel.push_back(listChannel.at(2));
            cv::Mat matOut;
            cv::merge(listOutChannel, matOut);
            cv::imshow("channel" + std::to_string(index), matOut);
        }
    }
    cv::waitKey(0);
#endif

}

void testBoxFilter()
{
    QString fileName1 = "D:/workspace/sample/lena(1).tiff";
    cv::Mat matSrc = cv::imread(fileName1.toStdString());

    cv::String windowName = "testBoxFilter";
    cvui::init(windowName);

    if(!matSrc.data)
    {
        qDebug() << __FILE__ << __LINE__
                 << "Failed to load image:" << fileName1;
        return;
    }

    cv::Mat dstMat;
    dstMat = cv::Mat::zeros(matSrc.size(), matSrc.type());
    cv::Mat windowMat = cv::Mat(cv::Size(dstMat.cols * 3, dstMat.rows),
                                matSrc.type());
    bool isBoxFilter = true;
    int ksize = 3;      // 核心大小
    int anchor = -1;    // 锚点, 正数的时候必须小于核心大小，即：-1 <= anchor < ksize
    cvui::window(windowMat, dstMat.cols, 0, dstMat.cols, dstMat.rows, "settings");
    while(true)
    {
        windowMat = cv::Scalar(0, 0, 0);
        // 原图先copy到左边
        cv::Mat leftMat = windowMat(cv::Range(0, matSrc.rows),
                                    cv::Range(0, matSrc.cols));
        cv::addWeighted(leftMat, 1.0f, matSrc, 1.0f, 0.0f, leftMat);
        // 中间为调整方框滤波参数的相关设置
        // 是否方框滤波
        cvui::checkbox(windowMat, 375, 10, "boxFilter", &isBoxFilter);
        cvui::printf(windowMat, 375, 40, "ksize");
        cvui::trackbar(windowMat, 375, 50, 165, &ksize, 1, 10);
        if(anchor >= ksize)
        {
            anchor = ksize - 1;
        }
        cvui::printf(windowMat, 375, 100, "anchor");
        cvui::trackbar(windowMat, 375, 110, 165, &anchor, -1, ksize-1);
        cv::boxFilter(matSrc,
                      dstMat,
                      -1,
                      cv::Size(ksize, ksize),
                      cv::Point(anchor, anchor),
                      isBoxFilter);
        // 效果图copy到右边
        // 注意：rang从位置1到位置2，不是位置1+宽度
        cv::Mat rightMat = windowMat(cv::Range(0, matSrc.rows),
                                     cv::Range(matSrc.cols * 2, matSrc.cols * 3));
        cv::addWeighted(rightMat, 0.0f, dstMat, 1.0f, 0.0f, rightMat);
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

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    testBoxFilter();
    return a.exec();
}

