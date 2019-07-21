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
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <fcntl.h>
#include <io.h>

using namespace std;
using namespace cv;

Kmeans::Kmeans(const shared_ptr<Data<float>> &data, int k, const Kmeans::TermCriteria &term_criteria,
               Kmeans::TYPE flag):data_(data),k_(k),term_criteria_(term_criteria),flag_(flag) {
}

bool Kmeans::InitCenters(shared_ptr<Data<float>> &centers) {
    memcpy(centers->data, data_->data, data_->cols*sizeof(float));

    float *distances = new float[data_->rows*k_];
    std::default_random_engine random_engine;
    random_engine.seed(1);
    vector<int> select_indexs;
    select_indexs.push_back(0);

    for(int i=1; i<k_; ++i){

        cv::Mat im_centers(i,centers->cols,CV_32FC1,centers->data);
        cv::Mat im_centers_T;
        cv::transpose(im_centers,im_centers_T);
        Math::Sgemm(data_->rows,data_->cols,i,data_->data, (float*)im_centers_T.data,distances);
        cv::Mat im_distance(data_->rows,i,CV_32FC1,distances);

        float tmp[data_->rows];
        for(int j=0; j<im_distance.rows; ++j) {
            float min_distance = im_distance.at<float>(j,0);

            for(int k=1; k<im_distance.cols; ++k) {
                if(min_distance>im_distance.at<float>(j,k)) {
                   min_distance = im_distance.at<float>(j,k);
                }
            }

            tmp[j] = 1 - min_distance;
            if (j) {
                tmp[j] += tmp[j-1];
            }
        }

        int select_index = 0;
        float sum = tmp[data_->rows-1];
        while(1) {
            float random_num = 1.0*random_engine()/random_engine.max();
            for(int j=0; j<data_->rows; ++j) {
                if (random_num<tmp[j]/sum) {
                    select_index = j;
                    break;
                }
            }

            bool flag = true;
            for(int j=0; j<select_indexs.size(); ++j) {
                if (select_indexs[j]==select_index) {
                    flag = false;
                    break;
                }
            }
            if(flag) {
                select_indexs.push_back(select_index);
                break;
            }
        }

        memcpy(centers->data+data_->cols*i, data_->data+select_index*data_->cols, data_->cols*sizeof(float));
        cout<<"init center: "<<i<<" success\tusing sample: "<<select_index<<endl;
    }

    delete(distances);
    return true;
}

void Kmeans::Cluster(shared_ptr<Data<int>> &labels, shared_ptr<Data<float>> &centers) {
    string center_file = "./centers.json.gz";
    if(access(center_file.c_str(),F_OK)) {
        if(!InitCenters(centers)) {
            cerr << "init centers failed"<<endl;
        }
        cv::Mat im_centers(centers->rows,centers->cols,CV_32FC1,centers->data);
        IOHelper::SaveMat2File(im_centers,center_file);
        cout<<"save center success"<<endl;
    } else {
        cv::Mat im_tmp;
        IOHelper::LoadMatFromFile(im_tmp,center_file);
        memcpy(centers->data,im_tmp.data,centers->rows*centers->cols* sizeof(float));
    }

    cv::Mat im_centers(centers->rows,centers->cols,CV_32FC1,centers->data);

    float *distances = new float[data_->rows*k_];
    for(int i=0; i<term_criteria_.iter; ++i) {
        float move_max_distance = 0;
        cv::Mat im_centers_T;
        cv::transpose(im_centers,im_centers_T);
        Math::Sgemm(data_->rows,data_->cols,centers->rows,data_->data, (float*)im_centers_T.data,distances);
        cv::Mat im_distance(data_->rows,centers->rows,CV_32FC1,distances);

        float tmp[data_->rows];
        int tmp_index[data_->rows];
        for(int j=0; j<im_distance.rows; ++j) {
            float min_distance = im_distance.at<float>(j,0);
            int min_index = 0;

            for(int k=1; k<im_distance.cols; ++k) {
                if(min_distance<im_distance.at<float>(j,k)) {
                    min_distance = im_distance.at<float>(j,k);
                    min_index = k;
                }
            }

            tmp[j] = min_distance;
            tmp_index[j] = min_index;
        }

        memcpy(labels->data, tmp_index, data_->rows*sizeof(int));

        vector<vector<int>> cluster_maps(k_);
        for(int j=0; j<data_->rows; ++j) {
            cluster_maps[tmp_index[j]].push_back(j);
        }

        for(int j=0; j<cluster_maps.size(); ++j) {
            if(cluster_maps[j].empty()){
                continue;
            }
            cv::Mat im(1,data_->cols,CV_32FC1,cv::Scalar(0));
            for(int k=0; k<cluster_maps[j].size(); ++k) {
                im += cv::Mat(1,data_->cols,CV_32FC1, data_->data+data_->cols*cluster_maps[j][k]);
            }

            im = im/cv::norm(im);
#if 0
            cv::Mat im_T;
            cv::transpose(im,im_T);
            cv::Mat d0;
            d0 = im*im_T;

            float test = d0.at<float>(0,0);
#endif

            cv::Mat center(centers->cols,1,CV_32FC1,centers->data+centers->cols*j);
            cv::Mat d1 = im*center;

            float distance = 1 - d1.at<float>(0,0);
            if(move_max_distance<distance) {
                move_max_distance = distance;
            }

            memcpy(centers->data+centers->cols*j, im.data,centers->cols*sizeof(float));
        }

        cout<<"iter "<<i<<" complete : move_max_distance : "<<move_max_distance<<endl;
        if(move_max_distance<term_criteria_.epsilon) {
            break;
        }
    }

    delete distances;

    IOHelper::SaveMat2File(im_centers,center_file);
    cv::Mat im_labels(labels->rows,labels->cols,CV_32FC1, labels->data);
    string label_file = "./labels.json.gz";
    IOHelper::SaveMat2File(im_labels,label_file);
}
