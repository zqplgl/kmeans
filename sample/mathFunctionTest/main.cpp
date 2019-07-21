//
// Created by zqp on 19-7-21.
//

#include <mathFunction.h>
#include <stdio.h>

void test_double(){
    int m = 5;
    int k = 8;
    int n = 2;

    double *a = new double[m*k];
    for(int i=0; i<m*k; ++i){
        a[i] = i*1.2;
    }
    double *b = new double[k*n];
    for(int i=0; i<k*n; ++i) {
        b[i] = i*5.6;
    }
    double *c = new double[m*n];
    Math::Dgemm(m,k,n,a,b,c);

    printf("\n****************a*****************\n");
    for(int i=0; i<m*k; ++i) {
        if(i%k==0) {
            printf("\n");
        }
        printf("%f\t", a[i]);
    }
    printf("\n****************a*****************\n");

    printf("\n****************b*****************\n");
    for(int i=0; i<k*n; ++i) {
        if(i%n==0) {
            printf("\n");
        }
        printf("%f\t", b[i]);
    }
    printf("\n****************b*****************\n");

    printf("\n****************c*****************\n");
    for(int i=0; i<m*n; ++i) {
        if(i%n==0) {
            printf("\n");
        }
        printf("%f\t", c[i]);
    }
    printf("\n****************c*****************\n");
}

void test_single(){
    int m = 5;
    int k = 8;
    int n = 2;

    float *a = new float[m*k];
    for(int i=0; i<m*k; ++i){
        a[i] = i*1.2;
    }
    float *b = new float[k*n];
    for(int i=0; i<k*n; ++i) {
        b[i] = i*5.6;
    }
    float *c = new float[m*n];
    Math::Sgemm(m,k,n,a,b,c);

    printf("\n****************a*****************\n");
    for(int i=0; i<m*k; ++i) {
        if(i%k==0) {
            printf("\n");
        }
        printf("%f\t", a[i]);
    }
    printf("\n****************a*****************\n");

    printf("\n****************b*****************\n");
    for(int i=0; i<k*n; ++i) {
        if(i%n==0) {
            printf("\n");
        }
        printf("%f\t", b[i]);
    }
    printf("\n****************b*****************\n");

    printf("\n****************c*****************\n");
    for(int i=0; i<m*n; ++i) {
        if(i%n==0) {
            printf("\n");
        }
        printf("%f\t", c[i]);
    }
    printf("\n****************c*****************\n");
}

int main() {
    test_single();
    test_double();
}