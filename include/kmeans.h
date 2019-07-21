//
// Created by zqp on 19-7-20.
//

#ifndef PROJECT_KMEANS_H
#define PROJECT_KMEANS_H

#include <memory>
template <typename T>
class Data {
public:
    Data(){}
    Data(int rows, int cols):rows(rows),cols(cols){data = new T[rows*cols];};
    Data(const Data& data) {
        rows = data.rows;
        cols = data.cols;
        this->data = new T[rows*cols];
        memcpy(data.data, this->data, rows*cols*sizeof(T));
    }
    ~Data() {
        if (data) {
            delete data;
        }
    }

public:
    int rows;
    int cols;
    T *data= nullptr;

};

class Kmeans {
public:
    struct TermCriteria {
        TermCriteria(){}
        TermCriteria(int type, int iter, double epsilon):type(type),iter(iter),epsilon(epsilon){}
        //1:epsilon     2:max_iter  3:epsilon+max_iter
        int type;
        int iter;
        double epsilon;
    };

    enum TYPE {
        KMEANS_PP_CENTERS = 0,
        KMEANS_RANDOM_CENTERS = 1,
    };

    Kmeans(const std::shared_ptr<Data<float>> &data, int k, const Kmeans::TermCriteria &term_criteria, Kmeans::TYPE flag);

    bool InitCenters(std::shared_ptr<Data<float>>& centers);

    void Cluster(std::shared_ptr<Data<int>> &labels, std::shared_ptr<Data<float>>& centers);

private:
    std::shared_ptr<Data<float>> data_;
    int k_;
    double *dev_data_;
    Kmeans::TermCriteria term_criteria_;
    Kmeans::TYPE flag_;

};

#endif //PROJECT_KMEANS_H
