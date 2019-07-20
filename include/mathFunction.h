//
// Created by zqp on 19-7-20.
//

#ifndef PROJECT_MATHFUNCTION_H
#define PROJECT_MATHFUNCTION_H

class Math  {
public:
    bool static Dgemm(int m, int k, int n, double *a, double *b, double *c);
    bool static Sgemm(int m, int k, int n, float *a, float *b, float *c);
};

#endif //PROJECT_MATHFUNCTION_H
