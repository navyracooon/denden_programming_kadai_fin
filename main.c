#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void load(const char *filename, int m, int n, float *A, float *b);

int main(int argc, char* argv[]) {
    float* A1 = malloc(sizeof(float) * 50 * 784);
    float* b1 = malloc(sizeof(float) * 50);
    float* A2 = malloc(sizeof(float) * 100 * 50);
    float* b2 = malloc(sizeof(float) * 100);
    float* A3 = malloc(sizeof(float) * 100 * 10);
    float* b3 = malloc(sizeof(float) * 10);
    float* x = load_mnist_bmp(argv[4]);
    load(argv[1], 50, 784, A1, b1);
    load(argv[2], 100, 50, A2, b2);
    load(argv[3], 10, 100, A3, b3);
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

void add(int m, const float* x, float* o) {
    for (int i=0; i<m; i++) {
        o[i] += x[i];
    }
}

void scale(int n, float x, float *o) {
    for (int i=0; i<n; i++) {
        o[i] *= x;
    }
}

void init(int n, float x, float *o) {
    for (int i=0; i<n; i++) {
        o[i] = x;
    }
}

void ordered_init(int n, int *o) {
    for (int i=0; i<n; i++) {
        o[i] = i;
    }
}

void rand_init(int n, float *o) {
    for (int i=0; i<n; i++) {
        o[i] = (((double)rand() / RAND_MAX) * 2) - 1;
    }
}

void swap(int *pa, int *pb) {
    int temp = *pa;
    *pa = *pb;
    *pb = temp;
}

void shuffle(int n, int *x) {
    for (int i=0; i<n; i++) {
        swap(&x[i], &x[rand() % n]);
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
        } else {
            y[i] = x[i];
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

int inference6(const float* A1, const float* b1,
               const float* A2, const float* b2,
               const float* A3, const float* b3,
               const float* x, float* y) {
    float ans;
    float* y1 = malloc(sizeof(float) * 50);
    float* y2 = malloc(sizeof(float) * 100);

    fc(50, 784, x, A1, b1, y1);
    relu(50, y1, y1);
    fc(100, 50, y1, A2, b2, y2);
    relu(100, y2, y2);
    fc(10, 100, y2, A3, b3, y);
    softmax(10, y, y);

    ans = max_index(10, y);
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

void backward6(const float* A1, const float* b1,
               const float* A2, const float* b2,
               const float* A3, const float* b3,
               const float*x, unsigned char t, float* y,
               float* dEdA1, float* dEdb1,
               float* dEdA2, float* dEdb2,
               float* dEdA3, float* dEdb3) {
    // 推論
    float* relu_x1 = malloc(sizeof(float) * 50);
    float* relu_x2 = malloc(sizeof(float) * 100);
    float* fc_x1 = malloc(sizeof(float) * 50);
    float* fc_x2 = malloc(sizeof(float) * 100);
    fc(50, 784, x, A1, b1, relu_x1);
    relu(50, relu_x1, fc_x1);
    fc(100, 50, fc_x1, A2, b2, relu_x2);
    relu(100, relu_x2, fc_x2);
    fc(10, 100, fc_x2, A3, b3, y);
    softmax(10, y, y);

    // 誤差逆伝播
    float* softmax_dEdx = malloc(sizeof(float) * 10);
    float* relu_dEdx2 = malloc(sizeof(float) * 100);
    float* relu_dEdx1 = malloc(sizeof(float) * 50);
    float* fc_dEdx3 = malloc(sizeof(float) * 100);
    float* fc_dEdx2 = malloc(sizeof(float) * 50);
    float* fc_dEdx1 = malloc(sizeof(float) * 784);

    softmaxwithloss_bwd(10, y, t, softmax_dEdx);
    fc_bwd(10, 100, fc_x2, softmax_dEdx, A3, dEdA3, dEdb3, fc_dEdx3);
    relu_bwd(100, relu_x2, fc_dEdx3, relu_dEdx2);
    fc_bwd(100, 50, fc_x1, relu_dEdx2, A2, dEdA2, dEdb2, fc_dEdx2);
    relu_bwd(50, relu_x1, fc_dEdx2, relu_dEdx1);
    fc_bwd(50, 784, x, relu_dEdx1, A1, dEdA1, dEdb1, fc_dEdx1);

    free(relu_x1);
    free(relu_x2);
    free(fc_x1);
    free(fc_x2);
    free(softmax_dEdx);
    free(relu_dEdx2);
    free(relu_dEdx1);
    free(fc_dEdx3);
    free(fc_dEdx2);
    free(fc_dEdx1);
}

float cross_entropy_error(const float* y, int t) {
    return -log(y[t] + 1e-7);
}

void save(const char *filename, int m, int n, const float *A, const float *b) {
    FILE *fp;
    fp = fopen(filename, "wb");
    fwrite(A, sizeof(float), m * n, fp);
    fwrite(b, sizeof(float), m, fp);
    fclose(fp);
}

void load(const char *filename, int m, int n, float *A, float *b) {
    FILE *fp;
    fp = fopen(filename, "rb");
    fread(A, sizeof(float), m * n, fp);
    fread(b, sizeof(float), m, fp);
    fclose(fp);
}
