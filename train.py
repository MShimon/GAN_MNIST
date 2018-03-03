import numpy as np
import tensorflow as tf
import os
from PIL import Image
#自作関数
from Discriminator import Discriminator
from Generator import Generator
import data

#画像のwidth,height
width = 28
height = 28
#hyper parametar
batch_size = 32
noise_dims = 1
iterations = 1000000
train_d = 3#generatorの学習一回に対して、discriminatorを学習させる回数
inter = 0.01

if __name__ == '__main__':
    # 必要な分だけメモリを確保するようにする
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    # 生成画像を保存する用のディレクトリ
    save_image_dir = "./generated_image"
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)

    #- 学習データの準備-#
    #MNISTデータをload
    x_train = data.load_MNIST()
    # batchの設定
    coord = tf.train.Coordinator()
    queue = tf.train.input_producer(x_train, shuffle=True)
    dequeue = queue.dequeue()
    batch = tf.train.batch([dequeue], batch_size=batch_size, allow_smaller_final_batch=True)

    #generator,discriminatorのインスタンス化
    generator = Generator(lr = 0.0001, weight_decay=0.01)
    discriminator = Discriminator(lr = 0.0001, weight_decay=0.01)

    #- generator,discriminatorの式 -#
    #placeholder
    _images_real = tf.placeholder(tf.float32, [None, width, height, 1])#入力画像用のplaceholder
    # 共通式
    noise = tf.random_uniform([batch_size, noise_dims])  # ノイズの取得
    images_fake = generator.forward(noise)  # ノイズからgeneratorが生成した画像
    output_real = discriminator.forward(_images_real)# 本物の画像を入力した時のdiscriminatorの出力
    output_fake = discriminator.forward(images_fake)# Generatorが出力した画像を入力した時のdiscriminatorの出力
    #generator,discriminatorのloss
    loss_D = discriminator.loss(output_fake, output_real)#
    loss_G = generator.loss(output_fake)
    # 更新式
    train_step_D = discriminator.train(loss_D)
    train_step_G = generator.train(loss_G)

    #- 学習途中での生成画像を保存するための処理 -#
    tmp_noise = np.arange(0.0, 1.0001, inter)#0〜1の一様分布を生成
    num_generated_image = tmp_noise.shape[0]  # 生成される画像数
    tmp_noise = tmp_noise.reshape(num_generated_image, 1).astype(np.float32)#今回のノイズは1次元
    uniform_noise = tf.constant(tmp_noise)
    image_fake_uniform = generator.forward(uniform_noise)# 一様分布から画像を生成

    #- 学習開始 -#
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        threads = tf.train.start_queue_runners(sess, coord)
        for i in range(iterations):  # 学習回数
            # - discriminatorの更新 -#
            for k in range(train_d):
                # - 実データの取得 -#
                images_real = sess.run(batch)
                # discriminatorの更新
                sess.run(train_step_D, feed_dict={_images_real: images_real})
            # - generatorの更新 -#
            sess.run(train_step_G)

            # 10000iteration毎に画像を保存
            if (i + 1) % 10000 == 0:
                #- 一様分布から画像を生成し、画像を保存する -#
                uni = sess.run(image_fake_uniform)
                #ディレクトリの作成
                save_epoch_dir = save_image_dir +"/epoch:" + str(i+1)
                if not os.path.exists(save_epoch_dir):
                    os.mkdir(save_epoch_dir)
                #一枚づつ画像を保存
                for i in range(num_generated_image):
                    value_noise = round(inter * i ,2)
                    res = uni[i].reshape(width,height)
                    pilImg = Image.fromarray(np.uint8(res*255))#255を掛けて正規化
                    file_name = str(value_noise) + ".png"  # ファイル名
                    pilImg.save(save_epoch_dir + "/" + file_name)