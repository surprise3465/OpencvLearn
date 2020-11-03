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
    QString fileName1 = FOLDERPATH + "blox.jpg";
    cv::Mat srcMat = cv::imread(fileName1.toStdString());
    cv::Mat dstMat;
    int width = 400;
    int height = 300;

    cv::resize(srcMat, srcMat, cv::Size(width, height));

    cv::String windowName = WINDOW_NAME.toStdString();
    cvui::init(windowName);

    cv::Mat windowMat = cv::Mat(cv::Size(srcMat.cols * 3,
                                         srcMat.rows * 4),
                                srcMat.type());
    int sigmaS = 100;
    int sigmaR = 1.0;

    int thresh = 232;
    int maxval = 255;

    while(true)
    {
        // 刷新全图黑色
        windowMat = cv::Scalar(0, 0, 0);

        // 原图复制
        cv::Mat mat = windowMat(cv::Range(srcMat.rows * 0, srcMat.rows * 1),
                                cv::Range(srcMat.cols * 0, srcMat.cols * 1));
        cv::addWeighted(mat, 0.0f, srcMat, 1.0f, 0.0f, mat);

        cv::Mat tempMat;
        {
            {
                cvui::printf(windowMat, 75 + width * 1, 40 + height * 0, "sigmaS");
                cvui::trackbar(windowMat, 75 + width * 1, 50 + height * 0, 165, &sigmaS, 101, 10000);
                cvui::printf(windowMat, 75 + width * 1, 90 + height * 0, "sigmaR");
                cvui::trackbar(windowMat, 75 + width * 1, 100, 165 + height * 0, &sigmaR, 1, 100);

                // 使用自适应流形应用高维滤波。
                cv::Ptr<cv::ximgproc::AdaptiveManifoldFilter> pAdaptiveManifoldFilter
                        = cv::ximgproc::createAMFilter(sigmaS/100.0f, sigmaR/100.0f, true);
                pAdaptiveManifoldFilter->filter(srcMat, tempMat);
                // 效果图copy
                mat = windowMat(cv::Range(srcMat.rows * 1, srcMat.rows * 2),
                                cv::Range(srcMat.cols * 0, srcMat.cols * 1));
                cv::addWeighted(mat, 0.0f, tempMat, 1.0f, 0.0f, mat);
            }

            //  转为灰度图像
            cv::cvtColor(tempMat, tempMat, cv::COLOR_BGR2GRAY);

            // 车牌时，对灰度图取反操作
//            tempMat = ~tempMat;

            {
                // 调整阈值化的参数thresh
                cvui::printf(windowMat, 75 + width * 1, 20 + height * 1, "thresh");
                cvui::trackbar(windowMat, 75 + width * 1, 40 + height * 1, 165, &thresh, 0, 255);
                // 调整阈值化的参数maxval
                cvui::printf(windowMat, 75 + width * 1, 80 + height * 1, "maxval");
                cvui::trackbar(windowMat, 75 + width * 1, 100 + height * 1, 165, &maxval, 0, 255);

                // 阈值化，注意：此处使用了THRESH_BINARY_INV，白色是255，255，255所以反转阈值化
                cv::threshold(tempMat, tempMat, thresh, maxval, cv::THRESH_BINARY_INV);
                // 效果图copy
                mat = windowMat(cv::Range(srcMat.rows * 0, srcMat.rows * 1),
                                cv::Range(srcMat.cols * 2, srcMat.cols * 3));

                //  转换图像
                cv::Mat grayMat;
                cv::cvtColor(tempMat, grayMat, cv::COLOR_GRAY2BGR);
                cv::addWeighted(mat, 0.0f, grayMat, 1.0f, 0.0f, mat);
            }

            // 寻找轮廓
            {
                std::vector<std::vector<cv::Point>> contours;
                std::vector<cv::Vec4i> hierarchy;
                // 查找轮廓：RETR_EXTERNAL-最外层轮廓
                cv::findContours(tempMat, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                // 遍历所有顶层轮廓，并绘制出来
                dstMat = srcMat.clone();
                cv::Mat emptyMat = srcMat.clone();
                emptyMat = cv::Scalar(0,0,0);

                // 拟合矩形框
                cv::Mat fittingRectMat = srcMat.clone();
                cv::Mat fittingMinAreaRectMat = srcMat.clone();
                cv::Mat fittingMinAreaCircleMat = srcMat.clone();
                cv::Mat fittingEllipseMat = srcMat.clone();
                cv::Mat fittingPolyMat = srcMat.clone();

                // 轮廓contours[i]对应4个hierarchy元素hierarchy[i][0]~ hierarchy[i][3]，
                // hierarchy[i][0]表示后一个轮廓的索引编号
                // hierarchy[i][1]前一个轮廓的索引编号
                // hierarchy[i][2]父轮廓的索引编号
                // hierarchy[i][3]内嵌轮廓的索引编号
                for(int index = 0; index >=0; index = hierarchy[index][0])
                {

                    if(hierarchy.size() <= 0)
                    {
                        break;
                    }
                    cv::Scalar color;
                    if(index < hierarchy.size() / 3)
                    {
                        color = cv::Scalar(250 / (hierarchy.size() / 3) * index, 125, 255);
                    }else if(index < hierarchy.size() / 3 * 2)
                    {
                        color = cv::Scalar(255, 250 / (hierarchy.size() / 3) * (index - hierarchy.size() / 3), 125);
                    }else
                    {
                        color = cv::Scalar(125, 255, 250 / (hierarchy.size() / 3 == 0 ? 1 :
                                                                              hierarchy.size() / 3) * (index - hierarchy.size() / 3 * 2));
                    }
                    // 绘制轮廓里面的第几个
                    cv::drawContours(emptyMat, contours, index, color, FILLED, 8, hierarchy);

                    // 寻找最大凸包
                    std::vector<cv::Point> hullPoints;
                    std::vector<int> hullIndex;
                    cv::convexHull(contours[index], hullPoints, false, true);
                    cv::convexHull(contours[index], hullIndex, false, false);
                    // 绘制凸包包围线
                    for(int index2 = 1; index2 < hullPoints.size(); index2++)
                    {
                        cv::line(mat, hullPoints.at(index2 - 1), hullPoints.at(index2), cv::Scalar(0, 0, 0), 2);
                        cv::line(dstMat, hullPoints.at(index2 - 1), hullPoints.at(index2), cv::Scalar(0, 0, 0), 2);
                    }
                    qDebug() << __FILE__ << __LINE__ << "index =" << index << "total =" << hierarchy.size();


                    // 使用形状拟合
                    // 使用外部包围矩形
                    {
                        cv::Rect rect = cv::boundingRect(hullPoints);
                        cv::rectangle(fittingRectMat, rect, cv::Scalar(0, 255, 0), 2);
                    }
                    // 使用外部最小包围矩形
                    {
                        cv::RotatedRect rotateRect = cv::minAreaRect(hullPoints);
                        cv::Point2f vertex[4];
                        rotateRect.points(vertex);
                        for(int index = 0; index < 4; index++)
                        {
                            cv::line(fittingMinAreaRectMat, vertex[index % 4], vertex[(index + 1) % 4], cv::Scalar(255, 0, 0), 2);
                        }
                    }
                    // 使用外部包围圆形（圆形就是最小了，不存在形变）
                    {
                        cv::Point2f center;
                        float radius;
                        cv::minEnclosingCircle(hullPoints, center, radius);
                        cv::circle(fittingMinAreaCircleMat, center, radius, cv::Scalar(0, 0, 0), 2);
                    }
                    // 使用外部椭圆拟合：至少要6个点
                    {
                        qDebug() << __FILE__ << __LINE__ << hullPoints.size();
                        if(hullPoints.size() >= 6)
                        {
                            cv::RotatedRect rotateRect = cv::fitEllipse(hullPoints);
                            cv::ellipse(fittingEllipseMat, rotateRect, cv::Scalar(0, 0, 0), 2);
                        }
                    }
                    // 使用多边形拟合
                    {
                        std::vector<cv::Point> polyPoints;
                        cv::approxPolyDP(hullPoints, polyPoints, 3, true);
                        std::vector<std::vector<cv::Point>> contour;
                        contour.push_back(polyPoints);
                        cv::drawContours(fittingPolyMat, contour, 0, cv::Scalar(0, 0, 0), 2);
                    }
                }

                // 效果图copy：轮廓图
                mat = windowMat(cv::Range(srcMat.rows * 1, srcMat.rows * 2),
                                cv::Range(srcMat.cols * 2, srcMat.cols * 3));
                cv::addWeighted(mat, 0.0f, emptyMat, 1.0f, 0.0f, mat);
                // 效果图copy：对已知轮廓进行最大凸包检测
                mat = windowMat(cv::Range(srcMat.rows * 2, srcMat.rows * 3),
                                cv::Range(srcMat.cols * 0, srcMat.cols * 1));
                cv::addWeighted(mat, 0.0f, dstMat, 1.0f, 0.0f, mat);
                // 效果图copy：黑色图拟合矩形
                mat = windowMat(cv::Range(srcMat.rows * 2, srcMat.rows * 3),
                                cv::Range(srcMat.cols * 1, srcMat.cols * 2));
                cv::addWeighted(mat, 0.0f, fittingRectMat, 1.0f, 0.0f, mat);
                // 效果图copy：原图拟合矩形
                mat = windowMat(cv::Range(srcMat.rows * 2, srcMat.rows * 3),
                                cv::Range(srcMat.cols * 1, srcMat.cols * 2));
                cv::addWeighted(mat, 0.0f, fittingRectMat, 1.0f, 0.0f, mat);
                // 效果图copy：原图拟合最小矩形
                mat = windowMat(cv::Range(srcMat.rows * 2, srcMat.rows * 3),
                                cv::Range(srcMat.cols * 2, srcMat.cols * 3));
                cv::addWeighted(mat, 0.0f, fittingMinAreaRectMat, 1.0f, 0.0f, mat);

                // 效果图copy：原图拟合最小圆形
                mat = windowMat(cv::Range(srcMat.rows * 3, srcMat.rows * 4),
                                cv::Range(srcMat.cols * 0, srcMat.cols * 1));
                cv::addWeighted(mat, 0.0f, fittingMinAreaCircleMat, 1.0f, 0.0f, mat);

                // 效果图copy：原图拟合最小椭圆
                mat = windowMat(cv::Range(srcMat.rows * 3, srcMat.rows * 4),
                                cv::Range(srcMat.cols * 1, srcMat.cols * 2));
                cv::addWeighted(mat, 0.0f, fittingEllipseMat, 1.0f, 0.0f, mat);

                // 效果图copy：原图拟合多边形
                mat = windowMat(cv::Range(srcMat.rows * 3, srcMat.rows * 4),
                                cv::Range(srcMat.cols * 2, srcMat.cols * 3));
                cv::addWeighted(mat, 0.0f, fittingPolyMat, 1.0f, 0.0f, mat);
            }
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
