#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void print(int m, int n, const float* x);
int inference3(const float* A, const float* b, const float* x);
void softmaxwithloss_bwd(int n, const float* y, unsigned char t, float* dEdx);
void relu_bwd(int n, const float* x, const float* dEdy, float* dEdx);

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
    int sum = 0;
    for (int i=0; i<test_count; i++) {
        if (inference3(A_784x10, b_784x10, test_x + i * width * height) == test_y[i]) {
            sum++;
        }
    }
    printf("%f%%\n", sum * 100.0 / test_count);
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

void relu(int n, const float* x, float* y) {
    for (int i=0; i<n; i++) {
        if (x[i] < 0) {
            y[i] = 0;
        }
    }
}

float max(int n, const float* x) {
    float max_val = x[0];
    for (int i=0; i<n; i++) {
        if (max_val < x[i]) {
            max_val = x[i];
        }
    }
    return max_val;
}

int max_index(int n, const float* x) {
    int max_index = 0;
    float max_val = x[0];
    for (int i=0; i<n; i++) {
        if (max_val < x[i]) {
            max_val = x[i];
            max_index = i;
        }
    }
    return max_index;
}

void softmax(int n, const float* x, float* y) {
    float sum_exp = 0;
    float max_val = max(n, x);
    for (int i=0; i<n; i++) {
        sum_exp += exp(x[i] - max_val);
    }
    for (int i=0; i<n; i++) {
        y[i] = exp(x[i] - max_val) / sum_exp;
    }
}

int inference3(const float* A, const float* b, const float* x) {
    int m = 10, n = 784;
    float ans;
    float* y = malloc(sizeof(float) * 10);

    fc(m, n, x, A, b, y);
    relu(m, y, y);
    softmax(m, y, y);

    ans = max_index(m, y);
    free(y);
    return ans;
}

void softmaxwithloss_bwd(int n, const float* y, unsigned char t, float* dEdx) {
    for (int i=0; i<n; i++) {
        if (i == t) {
            dEdx[i] = y[i] - 1;
        } else {
            dEdx[i] = y[i];
        }
    }
}

void relu_bwd(int n, const float* x, const float* dEdy, float* dEdx) {
    for (int i=0; i<n; i++) {
        if (x[i] > 0) {
            dEdx[i] = dEdy[i];
        } else {
            dEdx[i] = 0;
        }
    }
}
