# Visual Attribute Transfer through Deep Image Analogy tensorflow
Visual Attribute Transfer through Deep Image Analogyを実装してtwitter botにしたものです。

# 前提環境
## GPU
このプログラムはGPU環境を前提にしています。
よってGPU環境じゃない場合は適切に変更する必要があります。
tensorflowをインストール際はtensorflow-gpuをインストールしてください。
## VGG19
本プログラムはvgg19を転移学習で使用しているためVGG19のモデルをダウンロード
私は下記サイトよりimagenet-vgg-verydeep-19.matを入手しました。（何かあったとしても責任は負いかねます）
http://www.vlfeat.org/matconvnet/pretrained/

# usage
## VisualAttributeTransferの使用方法
任意のpythonプログラムに以下のような記述をすることで使用できます。
``` 
import VisualAttributeTransfer as VAT

pathes = ["image1.jpg", "image2.jpg"] #変換したい画像のペアのパスを用意します

ret = VAT.run(pathes)  # ここで変換

```
これでおそらく同じディレクトリに変換後の画像が2枚できます。

# 作成画像例
![sample](https://github.com/akikan/Visual_attribute_transfer_deep_image_analogy_tensorflow/blob/master/sample.png "sample")

## GPUを使えない環境な場合
1回の使用5時間ぐらいかかる可能性があります。

## bot.pyの設定
twitter　developperにアカウント登録してコンシューマーキーなどを持ってきます。
その後、bot.pyの15行目以降の""の部分に適切なものを入力します
