#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

void init(int n, float x, float *o);
void ordered_init(int n, int *o);
void rand_init(int n, float *o);
void shuffle(int n, int *x);
void add(int m, const float* x, float* o);
void scale(int n, float x, float *o);
int inference6(const float* A1, const float* b1,
               const float* A2, const float* b2,
               const float* A3, const float* b3,
               const float* x, float* y);
void backward6(const float* A1, const float* b1,
               const float* A2, const float* b2,
               const float* A3, const float* b3,
               const float*x, unsigned char t, float* y,
               float* dEdA1, float* dEdb1,
               float* dEdA2, float* dEdb2,
               float* dEdA3, float* dEdb3);
float cross_entropy_error(const float* y, int t);
void save(const char *filename, int m, int n, const float *A, const float *b);
void load(const char *filename, int m, int n, float *A, float *b);
bool compare_string(int n, char* a, char* b);

int main(int argc, char* argv[]) {

    // ファイルの読み書きの管理
    bool loadFile = false;
    bool saveFile = false;
    bool importFile = false;
    char* importFileName;
    if (argc > 1) {
        for (int i=1; i <= argc - 1; i++) {
            if (compare_string(2, "-l", argv[i])) {
                loadFile = true;
            } else if (compare_string(2, "-s", argv[i])) {
                saveFile = true;
            } else if (compare_string(2, "-i", argv[i])) {
                importFile = true;
            }
        }
        if (importFile) {
            importFileName = argv[argc - 1];
        }
    }

    // 学習・テストデータ読み込み
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
    float* A1 = malloc(sizeof(float) * 50 * 784);
    float* b1 = malloc(sizeof(float) * 50);
    float* A2 = malloc(sizeof(float) * 100 * 50);
    float* b2 = malloc(sizeof(float) * 100);
    float* A3 = malloc(sizeof(float) * 10 * 100);
    float* b3 = malloc(sizeof(float) * 10);
    if (!loadFile) {

        // 学習
        const int epoch = 10;
        const int n = 100;
        const int N = train_count;
        const float h = 0.1;

        rand_init(50 * 784, A1);
        rand_init(50, b1);
        rand_init(100 * 50, A2);
        rand_init(100, b2);
        rand_init(10 * 100, A3);
        rand_init(10, b3);

        // srand(time(NULL)) を入れた方が良いが、性能比較をしやすくするために抜いてある

        for (int i=0; i<epoch; i++) {
            int* index = malloc(sizeof(int) * N);  //高速化の観点からはmalloc宣言はループ外に出せる
            ordered_init(N, index);
            shuffle(N, index);

            int next_index = index[0];
            int offset = 0;
            for (int j=0; j<N/n; j++) {
                float* avg_dEdA1 = malloc(sizeof(float) * 50 * 784);
                float* avg_dEdb1 = malloc(sizeof(float) * 50);
                float* avg_dEdA2 = malloc(sizeof(float) * 100 * 50);
                float* avg_dEdb2 = malloc(sizeof(float) * 100);
                float* avg_dEdA3 = malloc(sizeof(float) * 10 * 100);
                float* avg_dEdb3 = malloc(sizeof(float) * 10);
                init(50 * 784, 0, avg_dEdA1);
                init(50, 0, avg_dEdb1);
                init(100 * 50, 0, avg_dEdA2);
                init(100, 0, avg_dEdb2);
                init(10 * 100, 0, avg_dEdA3);
                init(10, 0, avg_dEdb3);

                for (int k=0; k<n; k++) {
                    next_index = index[k + offset];

                    float* y = malloc(sizeof(float) * 10);
                    float* dEdA1 = malloc(sizeof(float) * 50 * 784);
                    float* dEdb1 = malloc(sizeof(float) * 50);
                    float* dEdA2 = malloc(sizeof(float) * 100 * 50);
                    float* dEdb2 = malloc(sizeof(float) * 100);
                    float* dEdA3 = malloc(sizeof(float) * 10 * 100);
                    float* dEdb3 = malloc(sizeof(float) * 10);

                    backward6(A1, b1, A2, b2, A3, b3,
                              train_x + 784 * next_index, train_y[next_index], y,
                              dEdA1, dEdb1, dEdA2, dEdb2, dEdA3, dEdb3);
                    add(50 * 784, dEdA1, avg_dEdA1);
                    add(50, dEdb1, avg_dEdb1);
                    add(100 * 50, dEdA2, avg_dEdA2);
                    add(100, dEdb2, avg_dEdb2);
                    add(10 * 100, dEdA3, avg_dEdA3);
                    add(10, dEdb3, avg_dEdb3);

                    free(y);
                    free(dEdA1);
                    free(dEdb1);
                    free(dEdA2);
                    free(dEdb2);
                    free(dEdA3);
                    free(dEdb3);
                }

                scale(50 * 784, (float) 1 / n, avg_dEdA1);
                scale(50, (float) 1 / n, avg_dEdb1);
                scale(100 * 50, (float) 1 / n, avg_dEdA2);
                scale(100, (float) 1 / n, avg_dEdb2);
                scale(10 * 100, (float) 1 / n, avg_dEdA3);
                scale(10, (float) 1 / n, avg_dEdb3);

                scale(50 * 784, -h, avg_dEdA1);
                scale(50, -h, avg_dEdb1);
                scale(100 * 50, -h, avg_dEdA2);
                scale(100, -h, avg_dEdb2);
                scale(10 * 100, -h, avg_dEdA3);
                scale(10, -h, avg_dEdb3);
                
                add(50 * 784, avg_dEdA1, A1);
                add(50, avg_dEdb1, b1);
                add(100 * 50, avg_dEdA2, A2);
                add(100, avg_dEdb2, b2);
                add(10 * 100, avg_dEdA3, A3);
                add(10, avg_dEdb3, b3);

                free(avg_dEdA1);
                free(avg_dEdb1);
                free(avg_dEdA2);
                free(avg_dEdb2);
                free(avg_dEdA3);
                free(avg_dEdb3);

                offset += n;
            }

            // 推論
            if (!importFile) {
                float E = 0;
                int correct_answer_sum = 0;

                for (int j=0; j < test_count; j++) {
                    float* y = malloc(sizeof(float) * 10);
                    if (inference6(A1, b1, A2, b2, A3, b3, test_x + j * width * height, y) == test_y[j]) {
                        correct_answer_sum++;
                    }
                    E += cross_entropy_error(y, test_y[j]);
                    free(y);
                }
                E /= test_count;
                float accuracy = ((float) correct_answer_sum / test_count) * 100;
                printf("Epoch: %d, Acc: %f%%, E: %f\n", i+1, accuracy, E);
            } else {
                float* x = load_mnist_bmp(importFileName);
                float* y = malloc(sizeof(float) * 10);
                int infer = inference6(A1, b1, A2, b2, A3, b3, x, y);
                printf("Epoch: %d | %s -> %d\n", i+1, importFileName, infer);
                free(y);

            }

            free(index);
        }
    } else {
        // パラメータファイル読み込み
        load("fc1.dat", 50, 784, A1, b1);
        load("fc2.dat", 100, 50, A2, b2);
        load("fc3.dat", 10, 100, A3, b3);

        // 推論
        if (!importFile) {
            float E = 0;
            int correct_answer_sum = 0;

            for (int j=0; j < test_count; j++) {
                float* y = malloc(sizeof(float) * 10);
                if (inference6(A1, b1, A2, b2, A3, b3, test_x + j * width * height, y) == test_y[j]) {
                    correct_answer_sum++;
                }
                E += cross_entropy_error(y, test_y[j]);
                free(y);
            }
            E /= test_count;
            float accuracy = ((float) correct_answer_sum / test_count) * 100;
            printf("Acc: %f%%, E: %f\n", accuracy, E);
        } else {
            float* x = load_mnist_bmp(importFileName);
            float* y = malloc(sizeof(float) * 10);
            int infer = inference6(A1, b1, A2, b2, A3, b3, x, y);
            printf("%s -> %d\n", importFileName, infer);
            free(y);

        }
    }

    if (saveFile) {
        save("fc1.dat", 50, 784, A1, b1);
        save("fc2.dat", 100, 50, A2, b2);
        save("fc3.dat", 10, 100, A3, b3);
    }

    free(A1);
    free(b1);
    free(A2);
    free(b2);
    free(A3);
    free(b3);
    return 0;
}

void print(int m, int n, const float* x) {
    /* 
     * m: 行の数
     * n: 列の数
     * x: 表示する要素を持つ浮動小数点数の配列へのポインタ
     */

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
    /* 
     * m: 要素の数
     * x: 加算される浮動小数点数の配列へのポインタ
     * o: 結果を格納する浮動小数点数の配列へのポインタ
     */

    for (int i=0; i<m; i++) {
        o[i] += x[i];
    }
}

void scale(int n, float x, float *o) {
    /* 
     * n: 要素の数
     * x: スケーリング因子（実数）
     * o: スケーリングが適用される浮動小数点数の配列へのポインタ
     */

    for (int i=0; i<n; i++) {
        o[i] *= x;
    }
}

void init(int n, float x, float *o) {
    /* 
     * n: 要素の数
     * x: 初期値（実数）
     * o: 初期化される浮動小数点数の配列へのポインタ
     */

    for (int i=0; i<n; i++) {
        o[i] = x;
    }
}

void ordered_init(int n, int *o) {
    /* 
     * n: 要素の数
     * o: 順番な整数を格納する配列へのポインタ
     */

    for (int i=0; i<n; i++) {
        o[i] = i;
    }
}

void rand_init(int n, float *o) {
    /* 
     * n: 要素の数
     * o: ランダムな値を格納する浮動小数点数の配列へのポインタ
     */

    for (int i=0; i<n; i++) {
        o[i] = (((double)rand() / RAND_MAX) * 2) - 1;
    }
}

void swap(int *pa, int *pb) {
    /* 
     * pa: 最初の交換要素へのポインタ
     * pb: 2番目の交換要素へのポインタ
     */

    int temp = *pa;
    *pa = *pb;
    *pb = temp;
}

void shuffle(int n, int *x) {
    /* 
     * n: 要素の数
     * x: シャッフルされる整数の配列へのポインタ
     */

    for (int i=0; i<n; i++) {
        swap(&x[i], &x[rand() % n]);
    }
}

void mul(int m, int n, const float* x, const float* A, float* y) {
    /* 
     * m: 行の数
     * n: 列の数
     * x: ベクトルxを表す浮動小数点数の配列へのポインタ
     * A: 行列Aを表す浮動小数点数の配列へのポインタ
     * y: ベクトルy（出力）を表す浮動小数点数の配列へのポインタ
     */

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
    /* 
     * m: 行の数
     * n: 列の数
     * x: ベクトルxを表す浮動小数点数の配列へのポインタ
     * A: 行列Aを表す浮動小数点数の配列へのポインタ
     * b: ベクトルbを表す浮動小数点数の配列へのポインタ
     * y: ベクトルy（出力）を表す浮動小数点数の配列へのポインタ
     */
    mul(m, n, x, A, y);
    add(m, b, y);
}

void relu(int n, const float* x, float* y) {
   /* 
    * n: 配列xの要素数
    * x: ReLU関数を適用する配列へのポインタ
    * y: ReLU関数の結果を格納する配列へのポインタ
    */

    for (int i=0; i<n; i++) {
        if (x[i] < 0) {
            y[i] = 0;
        } else {
            y[i] = x[i];
        }
    }
}

float max(int n, const float* x) {
    /* 
     * n: 配列xの要素数
     * x: 最大値を検索する配列へのポインタ
     * max_val: 最大値
     */

    float max_val = x[0];
    for (int i=0; i<n; i++) {
        if (max_val < x[i]) {
            max_val = x[i];
        }
    }
    return max_val;
}

int max_index(int n, const float* x) {
    /* 
     * n: 配列xの要素数
     * x: 最大値のインデックスを検索する配列へのポインタ
     * max_index: 最大値のインデックス
     */

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
    /* 
     * n: 配列xの要素数
     * x: ソフトマックス関数を適用する配列へのポインタ
     * y: ソフトマックス関数の結果を格納する配列へのポインタ
     */

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
    /* 
     * A1, A2, A3: それぞれの層の重みへのポインタ
     * b1, b2, b3: それぞれの層のバイアスへのポインタ
     * x: 入力データへのポインタ
     * y: 出力結果を格納する配列へのポインタ
     */
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
    /* 
     * n: 配列yと配列dEdxの要素数
     * y: ソフトマックス関数の出力へのポインタ
     * t: 教師ラベル
     * dEdx: 誤差逆伝播の結果を格納する配列へのポインタ
     */

    for (int i=0; i<n; i++) {
        if (i == t) {
            dEdx[i] = y[i] - 1;
        } else {
            dEdx[i] = y[i];
        }
    }
}

void relu_bwd(int n, const float* x, const float* dEdy, float* dEdx) {
    /* 
     * n: 配列x, 配列dEdy, 配列dEdxの要素数
     * x: ReLU関数の入力へのポインタ
     * dEdy: 前の層からの誤差へのポインタ
     * dEdx: 誤差逆伝播の結果を格納する配列へのポインタ
     */

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
    /* 
     * m: 行の数
     * n: 列の数
     * x: ベクトルxを表す浮動小数点数の配列へのポインタ
     * dEdy: 出力関連の誤差（error）を表す浮動小数点数の配列へのポインタ
     * A: 行列Aを表す浮動小数点数の配列へのポインタ
     * dEdA: 行列A関連の誤差（error）を表す浮動小数点数の配列へのポインタ
     * dEdb: ベクトルb関連の誤差（error）を表す浮動小数点数の配列へのポインタ
     * dEdx: ベクトルx関連の誤差（error）を表す浮動小数点数の配列へのポインタ
     */

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
    /* 
     * 各引数は全結合層やReLU関数、Softmax関数のパラメータと勾配
     * A1, A2, A3: 各全結合層の重み
     * b1, b2, b3: 各全結合層のバイアス
     * x: 入力データ
     * t: ターゲット（正解）データ
     * y: 出力データ
     * dEdA1, dEdA2, dEdA3: 各全結合層の重みに関する誤差
     * dEdb1, dEdb2, dEdb3: 各全結合層のバイアスに関する誤差
     */

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
    /* 
     * y: ニューラルネットワークの出力
     * t: ターゲット（正解）データ
     */

    return -log(y[t] + 1e-7);
}

void save(const char *filename, int m, int n, const float *A, const float *b) {
    /* 
     * filename: ファイル名
     * m: 行の数
     * n: 列の数
     * A: 保存する行列A
     * b: 保存するベクトルb
     */

    FILE *fp;
    fp = fopen(filename, "wb");
    fwrite(A, sizeof(float), m * n, fp);
    fwrite(b, sizeof(float), m, fp);
    fclose(fp);
}

void load(const char *filename, int m, int n, float *A, float *b) {
    /* 
     * filename: ファイル名
     * m: 行の数
     * n: 列の数
     * A: データをロードするための行列A
     * b: データをロードするためのベクトルb
     */

    FILE *fp;
    fp = fopen(filename, "rb");
    fread(A, sizeof(float), m * n, fp);
    fread(b, sizeof(float), m, fp);
    fclose(fp);
}

bool compare_string(int n, char* a, char* b) {
    /* 
     * n: 文字列の長さ
     * a, b: 比較する文字列
     */

    for (int i=0; i<n; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}
