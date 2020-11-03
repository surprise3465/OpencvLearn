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

QString WINDOW_NAME	= "YoloV3";
QString FOLDERPATH = "D:/workspace/sample/";


int main(int argc, char *argv[])
{
    std::string classesFile = (FOLDERPATH + "coco.names").toStdString();

    std::string modelWeights = (FOLDERPATH + "yolov3.weights").toStdString();

    std::string modelCfg = (FOLDERPATH + "yolov3.cfg").toStdString();

    std::ifstream ifs(classesFile);
    std::vector<std::string> classes;
    std::string classLine;
    while(std::getline(ifs, classLine))
    {
        classes.push_back(classLine);
    }

    // 加载yolov3模型
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelCfg, modelWeights);
    if(net.empty())
    {
        qDebug() << __FILE__ << __LINE__ << "net is empty!!!";
        return 0;
    }

    cv::Mat mat;
    cv::Mat blob;

    // 获得所有层的名称和索引
    std::vector<cv::String> layerNames = net.getLayerNames();
    int lastLayerId = net.getLayerId(layerNames[layerNames.size() - 1]);
    cv::Ptr<cv::dnn::Layer> lastLayer = net.getLayer(cv::dnn::DictValue(lastLayerId));
    qDebug() << __FILE__ << __LINE__
             << QString(lastLayer->type.c_str())
             << QString(lastLayer->getDefaultName().c_str())
             << QString(layerNames[layerNames.size()-1].c_str());

    // 获取输出的层
    std::vector<cv::String> outPutNames;
    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    for(int index = 0; index < outLayers.size(); index++)
    {
        outPutNames.push_back(layerNames[outLayers[index] - 1]);
        qDebug() << __FILE__ << __LINE__
                 << QString(layerNames[outLayers[index] - 1].c_str());
    }

    while(true)
    {
        // 读取图片识别
        mat = cv::imread((FOLDERPATH + "kite.jpg").toStdString());
        if(!mat.data)
        {
            qDebug() << __FILE__ << __LINE__ << "Failed to read image!!!";
            return 0;
        }

//        cv::dnn::blobFromImage(mat, blob);
        // 必须要设置，否则会跑飞
        cv::dnn::blobFromImage(mat,
                               blob,
                               1.0f/255,
                               cv::Size(320, 320),
                               cv::Scalar(0, 0, 0),
                               true,
                               false);
        net.setInput(blob);
        // 推理预测：可以输入预测的图层名称
        std::vector<cv::Mat> probs;
        net.forward(probs, outPutNames);

        // 显示识别花费的时间
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = cv::format("Inference time: %.2f ms", t);
        cv::putText(mat,
                  label,
                  cv::Point(0, 15),
                  cv::FONT_HERSHEY_SIMPLEX,
                  0.5,
                  cv::Scalar(255, 0, 0));
        // 置信度预制，大于执行度的将其使用rect框出来
        for(int index = 0; index < probs.size(); index++)
        {
            for (int row = 0; row < probs[index].rows; row++)
            {
                // 获取probs中一个元素里面匹配对的所有对象中得分最高的
                cv::Mat scores = probs[index].row(row).colRange(5, probs[index].cols);
                cv::Point classIdPoint;
                double confidence;
                // Get the value and location of the maximum score
                cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if(confidence > 0.6)
                {
                    qDebug() << __FILE__ << __LINE__ << confidence << classIdPoint.x;
                    int centerX = (int)(probs.at(index).at<float>(row, 0) * mat.cols);
                    int centerY = (int)(probs.at(index).at<float>(row, 1) * mat.rows);
                    int width   = (int)(probs.at(index).at<float>(row, 2) * mat.cols);
                    int height  = (int)(probs.at(index).at<float>(row, 3) * mat.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    cv::Rect objectRect(left, top, width, height);
                    cv::rectangle(mat, objectRect, cv::Scalar(255, 0, 0), 2);
                    cv::String label = cv::format("%s:%.4f",
                                                  classes[classIdPoint.x].data(),
                                                  confidence);
                    cv::putText(mat,
                                label,
                                cv::Point(left, top - 10),
                                cv::FONT_HERSHEY_SIMPLEX,
                                0.4,
                                cv::Scalar(0, 0, 255));
                    qDebug() << __FILE__ << __LINE__
                            << centerX << centerY << width << height;
                }
            }
        }

        cv::imshow(WINDOW_NAME.toStdString(), mat);
        cv::waitKey(0);
    }

}
