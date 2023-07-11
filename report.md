# 序章
## このレポートについて
このレポートは`.md`(マークダウン形式)で書かれております。
また、このレポートと総合課題のソースコードはすべてGitで管理され、[このリポジトリ](https://github.com/navyracooon/denden_programming_kadai_fin)にて公開されています。
さらに、最終レポート及び全ての補題はGitの一つのコミットに対応しております。

# 本論
## 概要
### 動かし方
このプログラムは以下のコマンドライン引数を受付けます、各オプションはすべて任意で、`-ls`のように短縮することは意図されておらず、全てスペース区切りで受付けます。

- `-l` -> Load Fileの意で、パラメータを一から作成せずに、パラメータファイルを読み込みます。`fc1.dat`、`fc2.dat`、`fc3.dat`がない場合にはエラーを吐きます。
- `-s` -> Save Fileの意で、パラメータファイルを出力します。命名は上記の通りです。
- `-i` -> Import Fileの意で、BMPファイルを受付ます。尚、このパラメータを指定した時には最後のオプションとしてファイル名を末尾に付け加えることが必須です。

### 処理の流れ
main.cのおおまかな動作について説明します。
1. ファイルの読み書きの管理。前項で述べたパラメータの解析をします。
2. 学習およびテストデータの読み込みをします。資料と同じです。
3. 処理層、`1.`の結果によって処理が変わります。
    3.1. パラメータファイルを読み込まない場合はまず学習から始めます。その後、importされたBMPファイルがあるならそれを、なければテストケースに対しての推論を行います。
    3.2. パラメータファイルを読み込む場合は推論から始めます。推論の流れは上記に述べたとおりです。
4. `-s`が選択されている場合はパラメータファイルに今回得られたパラメータを保存します。
5. 最後にメモリの解放を行って終わりです。

一例として、何もパラメータを指定しない場合、以下のような結果が得られます。
```
Epoch: 1, Acc: 62.750000%, E: 1.094088
Epoch: 2, Acc: 78.619995%, E: 0.697560
Epoch: 3, Acc: 82.500000%, E: 0.566385
Epoch: 4, Acc: 85.389999%, E: 0.482162
Epoch: 5, Acc: 86.610001%, E: 0.442749
Epoch: 6, Acc: 88.040001%, E: 0.400204
Epoch: 7, Acc: 88.550003%, E: 0.387930
Epoch: 8, Acc: 88.739998%, E: 0.369683
Epoch: 9, Acc: 89.590004%, E: 0.345224
Epoch: 10, Acc: 90.070000%, E: 0.346835
```


## 関数の説明

ほとんどの関数は特段説明することがないため、主要な2つの関数について説明します

``` inference6
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
```

推論を行う関数となります。非常に単純で、5層の全結合層、ReLU関数、Softmax関数を順番に通して、一番確率が高いものを返すだけの関数となっています。

``` backward6
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
```
今回の中で一番長い関数となります。誤差逆伝播を担当する箇所であり、仕組みとしては
1. メモリを確保
2. 順番に推論を行う
3. `2.`の推論の誤差逆伝播処理を行う
4. メモリ解放
の4つの仕組みからなっています。
それぞれの層における誤差逆伝播の仕組みは資料の通りで、特別な処理は行っていません。

## 工夫した点等
### 工夫した点
- 6層としての実装を行いました
- コマンドとして使いやすいような配慮をしました
- できる限り可読性を意識したコードを作成しました (そのため冗長な表現も何点か存在します)
- srandを固定することで、パラメータを変えた時の振る舞いの違いが観察しやすいようにしました
### 工夫しようとして失敗した点
- パラメータチューニングをしっかりやってみようと思っていたのですが、一度の学習時間がそれなりにかかってしまうこともあり、優位なパラメータを見つけられませんでした。

# 終章
結果として、精度90%のモデルが作れたのは非常に興味深かったです。
以前も個人的にKerasやPyTorchなどでMNISTは学習させていましたが、各層の詳しい働きについては学習が甘い点があったため、今回綺麗に理解できてよかったです。
反省点としては、うまくパラメータチューニングができなかった点にあり、今度とも個人的に学習を続けてみたく思いました。
