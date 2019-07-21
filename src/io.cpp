//
// Created by zqp on 19-7-21.
//
#include <io.h>

bool IOHelper::SaveMat2File(const cv::Mat &im, const std::string &file_path) {
    cv::FileStorage fs(file_path, cv::FileStorage::WRITE);
    fs.write("mat", im);
    fs.release();
    return true;
}

bool IOHelper::LoadMatFromFile(cv::Mat &im, const std::string &file_path) {
    cv::FileStorage fs(file_path,cv::FileStorage::READ);
    fs["mat"]>>im;
    fs.release();
    return true;
}


