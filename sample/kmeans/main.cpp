//
// Created by zqp on 19-7-21.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <kmeans.h>
#include <memory>
using namespace std;

int main() {
    string json_path = "./features1.json.gz";
    cv::FileStorage fs1(json_path, cv::FileStorage::READ);
    cv::Mat features;
    fs1["features"]>>features;
    cerr << "loading features1 data success" << endl;

    cerr << "features info:\n\trows: "<<features.rows<<"\n\tcols: "<<features.cols<<endl;

    shared_ptr<Data<float>> data_ptr = make_shared<Data<float>>(features.rows,features.cols);
    memcpy(data_ptr->data, features.data, features.rows*features.cols*sizeof(float));
    int k = 1000;
    Kmeans::TermCriteria term_criteria(3, 1000,0.1);
    Kmeans kmeans(data_ptr,k, term_criteria,Kmeans::TYPE::KMEANS_PP_CENTERS);


}

