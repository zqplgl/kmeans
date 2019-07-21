//
// Created by zqp on 19-7-21.
//

#ifndef PROJECT_IO_H
#define PROJECT_IO_H

#include <opencv2/opencv.hpp>

class IOHelper{
public:
    bool static SaveMat2File(const cv::Mat& im, const std::string& file_path);
    bool static LoadMatFromFile(cv::Mat& im, const std::string& file_path);

};


#endif //PROJECT_IO_H
