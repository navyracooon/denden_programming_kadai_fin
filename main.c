#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void init(int n, float x, float *o);
void ordered_init(int n, int *o);
void rand_init(int n, float *o);
void shuffle(int n, int *x);
void add(int m, const float* x, float* o);
void scale(int n, float x, float *o);
int inference3(const float* A, const float* b, const float* x, float* y);
void backward3(const float* A, const float* b, const float*x, unsigned char t,
               float* y, float* dEdA, float* dEdb);
float cross_entropy_error(const float* y, int t);

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
    int epoch = 10;
    int n = 100;
    int N = train_count;
    float h = 0.1;

    float* A = malloc(sizeof(float) * 10 * 784);
    float* b = malloc(sizeof(float) * 10);
    rand_init(10 * 784, A);
    rand_init(10, b);


    for (int i=0; i<epoch; i++) {
        int* index = malloc(sizeof(int) * N);  //高速化の観点からはmalloc宣言はループ外に出せる
        ordered_init(N, index);
        shuffle(N, index);

        int next_index = index[0];
        int offset = 0;
        for (int j=0; j<N/n; j++) {
            float* avg_dEdA = malloc(sizeof(float) * 10 * 784);
            float* avg_dEdb = malloc(sizeof(float) * 10);
            init(10 * 784, 0, avg_dEdA);
            init(10, 0, avg_dEdb);

            for (int k=0; k<n; k++) {
                next_index = index[k + offset];

                float* y = malloc(sizeof(float) * 10);
                float* dEdA = malloc(sizeof(float) * 10 * 784);
                float* dEdb = malloc(sizeof(float) * 10);
                backward3(A, b, train_x + 784 * next_index, train_y[next_index], y, dEdA, dEdb);
                add(10 * 784, dEdA, avg_dEdA);
                add(10, dEdb, avg_dEdb);

                free(y);
                free(dEdA);
                free(dEdb);
            }

            scale(10 * 784, (float) 1 / n, avg_dEdA);
            scale(10, (float) 1 / n, avg_dEdb);

            scale(10 * 784, -h, avg_dEdA);
            scale(10, -h, avg_dEdb);
            
            add(10 * 784, avg_dEdA, A);
            add(10, avg_dEdb, b);

            free(avg_dEdA);
            free(avg_dEdb);

            offset += n;
        }

        float E = 0;
        int correct_answer_sum = 0;

        for (int j=0; j < test_count; j++) {
            float* y = malloc(sizeof(float) * 10);
            if (inference3(A, b, test_x + j * width * height, y) == test_y[j]) {
                correct_answer_sum++;
            }
            E += cross_entropy_error(y, test_y[j]);
            free(y);
        }
        E /= test_count;
        float accuracy = ((float) correct_answer_sum / test_count) * 100;

        printf("Epoch: %d, Acc: %f%%, E: %f\n", i+1, accuracy, E);
        free(index);
    }
    free(A);
    free(b);
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

int inference3(const float* A, const float* b, const float* x, float* y) {
    float ans;

    fc(10, 784, x, A, b, y);
    relu(10, y, y);
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

float cross_entropy_error(const float* y, int t) {
    return -log(y[t] + 1e-7);
}

