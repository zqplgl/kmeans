//
// Created by zqp on 19-7-20.
//
#include <kmeans.h>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>
#include <random>
#include <string.h>
#include <mathFunction.h>

using namespace std;

Kmeans::Kmeans(const shared_ptr<Data<double>> &data, int k, const Kmeans::TermCriteria &term_criteria,
               Kmeans::TYPE flag):data_(data),k_(k),term_criteria_(term_criteria),flag_(flag) {
//    cudaError_t cuda_stat;
//
//    cuda_stat = cudaMalloc((void**)&dev_data_, data->rows*data->cols*sizeof(double));
//    if (cuda_stat != cudaSuccess) {
//        cerr<<"device memory allocation failed"<<endl;
//        exit(0);
//    }
//
//    cuda_stat = cudaMemcpy(dev_data_, data->data, data->rows*data->cols* sizeof(double), cudaMemcpyHostToDevice);
//    if(cuda_stat!=cudaSuccess) {
//        cudaFree(dev_data_);
//        cerr<<"data download failed"<<endl;
//        exit(0);
//    }
}

bool Kmeans::InitCenters(shared_ptr<Data<double>> &centers) {
    memcpy(centers->data, data_->data, data_->cols*sizeof(double));

    double *distances = new double[data_->rows*k_];
    for(int i=1; i<k_; ++i){
        Math::Dgemm(data_->rows,data_->cols,i,data_->data, centers->data,distances);
    }
    return true;

}

void Kmeans::Cluster(shared_ptr<Data<double>> &labels, shared_ptr<Data<double>> &centers) {
    if(!InitCenters(centers)) {
        cerr << "init centers failed"<<endl;
    }

}

int main(){
    cout<<"hello world"<<endl;
    return 0;
}
