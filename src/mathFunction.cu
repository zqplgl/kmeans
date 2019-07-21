//
// Created by zqp on 19-7-20.
//
#include <mathFunction.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <iostream>

using namespace std;

bool Math::Dgemm(int m, int k, int n, double *a, double *b, double *c) {
    double *dev_a, *dev_b, *dev_c;
    cublasStatus_t stat;
    cudaError_t stat_a, stat_b, stat_c;

    stat_a = cudaMalloc((void**)&dev_a, m*k*sizeof(double));
    stat_b = cudaMalloc((void**)&dev_b, k*n*sizeof(double));
    stat_c = cudaMalloc((void**)&dev_c, m*n*sizeof(double));
    if (stat_a != cudaSuccess || stat_b!=cudaSuccess || stat_c!=cudaSuccess) {
        if(stat_a==cudaSuccess)
            cudaFree(dev_a);
        if(stat_b==cudaSuccess)
            cudaFree(dev_b);
        if(stat_c==cudaSuccess)
            cudaFree(dev_c);
        printf("device memory allocation failed\n");
        return false;
    }

    stat_a = cudaMemcpy(dev_a, a, m*k*sizeof(double), cudaMemcpyHostToDevice);
    stat_b = cudaMemcpy(dev_b, b, k*n*sizeof(double), cudaMemcpyHostToDevice);
    if (stat_a!=cudaSuccess || stat_b!=cudaSuccess) {
        printf("data download failed\n");
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        return false;
    }

    cublasHandle_t handle;
    stat = cublasCreate_v2(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        return false;
    }

    double alpha = 1.0;
    double beta = 0.0;

    cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dev_b, n, dev_a, k, &beta, dev_c, n);

    stat_c = cudaMemcpy(c,dev_c,m*n*sizeof(double),cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cublasDestroy_v2(handle);

    if(stat_c!=cudaSuccess){
        printf("copy dev_c--->c failed\n");
        return false;
    }

    return true;
}

bool Math::Sgemm(int m, int k, int n, float *a, float *b, float *c) {
    float *dev_a, *dev_b, *dev_c;
    cublasStatus_t stat;
    cudaError_t stat_a, stat_b, stat_c;

    stat_a = cudaMalloc((void**)&dev_a, m*k*sizeof(float));
    stat_b = cudaMalloc((void**)&dev_b, k*n*sizeof(float));
    stat_c = cudaMalloc((void**)&dev_c, m*n*sizeof(float));
    if (stat_a != cudaSuccess || stat_b!=cudaSuccess || stat_c!=cudaSuccess) {
        if(stat_a==cudaSuccess)
            cudaFree(dev_a);
        if(stat_b==cudaSuccess)
            cudaFree(dev_b);
        if(stat_c==cudaSuccess)
            cudaFree(dev_c);
        printf("device memory allocation failed\n");
        return false;
    }

    stat_a = cudaMemcpy(dev_a, a, m*k*sizeof(float), cudaMemcpyHostToDevice);
    stat_b = cudaMemcpy(dev_b, b, k*n*sizeof(float), cudaMemcpyHostToDevice);
    if (stat_a!=cudaSuccess || stat_b!=cudaSuccess) {
        printf("data download failed\n");
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        return false;
    }

    cublasHandle_t handle;
    stat = cublasCreate_v2(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        return false;
    }

    float alpha = 1.0;
    float beta = 0.0;

    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dev_b, n, dev_a, k, &beta, dev_c, n);

    stat_c = cudaMemcpy(c,dev_c,m*n*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cublasDestroy_v2(handle);

    if(stat_c!=cudaSuccess){
        printf("copy dev_c--->c failed\n");
        return false;
    }

    return true;
}




