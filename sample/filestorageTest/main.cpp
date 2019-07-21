//
// Created by zqp on 19-7-21.
//
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
int main() {
    string json_path = "./test.json.gz";
    cv::FileStorage fs(json_path, cv::FileStorage::WRITE);
    cv::Mat im = cv::imread("/home/zqp/picture/7.jpg", 0);
    cv::imshow("im",im);

    fs << "num"<<12;
    fs << "height"<<15;
    fs << "im"<<im;
    fs.release();
}

