//
// Created by zqp on 19-7-21.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <kmeans.h>
#include <memory>
#include <io.h>
using namespace std;

int main(int argc, char** argv) {
    if(argc!=5) {
        cerr<<"useage: kmeans feature_path k iters epsilon"<<endl;
        return 0;
    }

    string json_path = argv[1];
    int k = stoi(argv[2]);
    int iters = stoi(argv[3]);
    double epsilon = stod(argv[4]);

    cv::Mat features;
    IOHelper::LoadMatFromFile(features,json_path);
    cerr << "loading features1 data success" << endl;

    cerr << "features info:\trows "<<features.rows<<"\tcols "<<features.cols<<endl;

    shared_ptr<Data<float>> data_ptr = make_shared<Data<float>>(features.rows,features.cols);
    memcpy(data_ptr->data, features.data, features.rows*features.cols*sizeof(float));

    Kmeans::TermCriteria term_criteria(3, iters,epsilon);
    Kmeans kmeans(data_ptr,k, term_criteria,Kmeans::TYPE::KMEANS_PP_CENTERS);
    shared_ptr<Data<int>> labels = make_shared<Data<int>>(features.rows,1);
    shared_ptr<Data<float>> centers = make_shared<Data<float>>(k,features.cols);
    kmeans.Cluster(labels,centers);


}

