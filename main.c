#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void print(int m, int n, const float* x);
void fc(int m, int n, const float* x, const float* A, const float* b, float* y);
void relu(int n, float* x, float* y);
void softmax(int n, float* x, float* y);

int main() {
    // データ読み込み
    float* train_x = NULL;
    unsigned char* train_y = NULL;
    int train_count = -1;

    float* test_x = NULL;
    unsigned char* test_y = NULL;
    int test_count = -1;

    int width = -1;
    int height = -1;

    load_mnist(&train_x, &train_y, &train_count,
               &test_x, &test_y, &test_count,
               &width, &height);

    // 処理層

    float* y = malloc(sizeof(float)*10);

    fc(10, 784, train_x, A_784x10, b_784x10, y);
    relu(10, y, y);
    print(1, 10, y);
    return 0;
}

void print(int m, int n, const float* x) {
    int offset = 0;
    for (int i=0; i<m; i++) {
        for (int j=0+offset; j<n+offset; j++) {
            printf("%lf ", x[j]);
        }
        offset += n;
    }
    printf("\n");
}

void add(int m, const float* b, float* y) {
    for (int i=0; i<m; i++) {
        y[i] += b[i];
    }
}

void mul(int m, int n, const float* x, const float* A, float* y) {
    float sum;
    int offset = 0;

    for (int i=0; i<m; i++) {
        sum = 0;
        for (int j=0; j<n; j++) {
            sum += (A[j + offset] * x[j]);
        }
        offset += n;
        y[i] = sum;
    }
}

void fc(int m, int n, const float* x, const float* A, const float* b, float* y) {
    mul(m, n, x, A, y);
    add(m, b, y);
}

void relu(int n, float* x, float* y) {
    for (int i=0; i<n; i++) {
        if (x[i] < 0) {
            y[i] = 0;
        }
    }
}

void softmax(int n, float* x, float* y) {
    int sum_exp = 0;
    for (int i=0; i<n; i++) {
        sum_exp += exp(x[i]);
    }
    for (int i=0; i<n; i++) {
        y[i] = exp(x[i]) / exp(sum_exp);
    }
}
