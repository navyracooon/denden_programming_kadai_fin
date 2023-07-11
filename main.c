#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void print(int m, int n, const float* x);
int inference3(const float* A, const float* b, const float* x);
void backward3(const float* A, const float* b, const float*x, unsigned char t,
               float* y, float* dEdA, float* dEdb);
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
    float* y = malloc(sizeof(float) * 10);
    float* dEdA = malloc(sizeof(float) * 784 * 10);
    float* dEdb = malloc(sizeof(float) * 10);
    backward3(A_784x10, b_784x10, train_x + 784 * 8, train_y[8], y, dEdA, dEdb);
    print(10, 784, dEdA);
    print(1, 10, dEdb);
    free(y);
    free(dEdA);
    free(dEdb);
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
    float ans;
    float* y = malloc(sizeof(float) * 10);

    fc(10, 784, x, A, b, y);
    relu(10, y, y);
    softmax(10, y, y);

    ans = max_index(10, y);
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

void fc_bwd(int m, int n, const float* x, const float* dEdy,
            const float* A, float* dEdA, float* dEdb, float* dEdx) {
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            dEdA[j + n * i] = dEdy[i] * x[j];
        }
        for (int i=0; i<m; i++) {
            dEdb[i] = dEdy[i];
        }
        for (int i=0; i<n; i++) {
            dEdx[i] = 0;
            for (int j=0; j<m; j++) {
                dEdx[i] += A[i + n * j] * dEdy[j];
            }
        }
    }
}

void backward3(const float* A, const float* b, const float*x, unsigned char t,
               float* y, float* dEdA, float* dEdb) {
    // 推論
    float* relu_x = malloc(sizeof(float) * 10);
    fc(10, 784, x, A, b, relu_x);
    relu(10, relu_x, y);
    softmax(10, y, y);

    // 誤差逆伝播
    float* softmax_dEdx = malloc(sizeof(float) * 10);
    float* relu_dEdx = malloc(sizeof(float) * 10);
    float* fc_dEdx = malloc(sizeof(float) * 784);

    softmaxwithloss_bwd(10, y, t, softmax_dEdx);
    relu_bwd(10, relu_x, softmax_dEdx, relu_dEdx);
    fc_bwd(10, 784, x, relu_dEdx, A, dEdA, dEdb, fc_dEdx);

    free(relu_x);
    free(softmax_dEdx);
    free(relu_dEdx);
    free(fc_dEdx);
}
