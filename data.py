import numpy as np
import matplotlib as plt
from keras.datasets import mnist
from PIL import Image

#@brief:MNISTのデータセットを読み込む
#@return:MNISTの訓練画像のみ 0〜1で正規化 shape(60000,28,28,1)
def load_MNIST():
    # データのダウンロード
    # 画像データはnp-array,正解ラベルはone-hotラベルで10クラス(0-9)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # データを0〜1で正規化
    x_train = x_train/255
    # 画像サイズを取得
    img_num, width, height = x_train.shape[0], x_train.shape[1], x_train.shape[2]
    # tensorflow用にshapeを変更
    images = x_train.reshape(img_num, width, height, 1).astype(np.float32)

    #今回は訓練画像だけ欲しい
    return images

#@brief:複数枚の画像を並べて表示する
#@param:digits  表示したい画像のリスト
#       num_width   横に並べる数
#       num_height  縦に並べる数
#       mode    show:画像を表示する, save:画像を保存する
#
def visualize_image(digits, num_width, num_height,mode = None,dir = None):
    for h in range(num_height):
        for w in range(num_width):
            # 画像を水平方向に連結していく
            if w != 0:
                tmp_img = np.hstack((tmp_img, digits[w + h * num_width]))
            else:
                tmp_img = digits[w + h * num_width]

        # 画像を垂直方向に連結する
        if h != 0:
            img = np.vstack((img, tmp_img))
        else:
            img = tmp_img

    #表示モード
    if mode == "show":
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()
    #保存モード
    elif mode == "save":
        if not dir == None:#ファイル名がしていされていれば
            pilImg = Image.fromarray(img)  # uintに変換
            pilImg.save(dir)

