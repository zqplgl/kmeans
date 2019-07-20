//
// Created by zqp on 19-7-20.
//
#include <kmeans.h>
#include <iostream>
//#include <cublas_v2.h>
//#include <cuda_runtime.h>
#include <memory>
#include <random>
#include <string.h>

#define IDX2C(i,j,ld) (((i)*(ld))+(j))

using namespace std;

Kmeans::Kmeans(const Data<double> &data, int k, const Kmeans::TermCriteria &term_criteria,
               Kmeans::TYPE flag) {
    data_ = data;
    k_ = k;
    term_criteria_ = term_criteria;
    flag_ = flag;
}

bool Kmeans::InitCenters(Data<double> &centers) {

    memcpy(centers.data, data_.data, centers.cols* sizeof(double));

    return true;

}

void shared_test(shared_ptr<double> a) {

}

void Kmeans::Cluster(Data<double> &labels, Data<double> &centers) {
    labels = Data<double>(data_.rows,2);
    centers = Data<double>(k_,data_.cols);

    if(!InitCenters(centers)) {
        cerr << "init centers failed"<<endl;
    }

}

int main(){
    cout<<"hello world"<<endl;
    return 0;
}
