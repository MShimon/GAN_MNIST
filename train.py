import numpy as np
import tensorflow as tf
import os
#自作関数
from Discriminator import Discriminator
from Generator import Generator
import data

#画像のwidth,height
width = 28
height = 28
#hyper parametar
save_width = 4
save_height = 8
batch_size = save_width * save_height#バッチサイズ
noise_dims = 1
iterations = 1000
train_d = 1#generatorの学習一回に対して、discriminatorを学習させる回数
#画像を保存するディレクトリ
save_image_dir = "./generated_image"
#tensorboardのlogを吐くディレクトリ
log_dir = "gan_mnist"

if __name__ == '__main__':
    # 必要な分だけメモリを確保するようにする
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    # 生成画像を保存する用のディレクトリ
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
    noise = tf.random_normal([batch_size, noise_dims])  # ノイズの取得
    images_fake = generator.forward(noise)  # ノイズからgeneratorが生成した画像
    output_real = discriminator.forward(_images_real)# 本物の画像を入力した時のdiscriminatorの出力
    output_fake = discriminator.forward(images_fake)# Generatorが出力した画像を入力した時のdiscriminatorの出力
    #generator,discriminatorのloss
    loss_D = discriminator.loss(output_fake, output_real)#
    loss_G = generator.loss(output_fake)
    # 更新式
    train_step_D = discriminator.train(loss_D)
    train_step_G = generator.train(loss_G)

    #tensorboardに記述
    tf.summary.scalar("loss_discriminator",loss_D)
    tf.summary.scalar("loss_generator",loss_G)
    summary = tf.summary.merge_all()

    #- 学習開始 -#
    with tf.Session() as sess:
        #TensorBoardの設定
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir, sess.graph_def)
        #初期化
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        threads = tf.train.start_queue_runners(sess, coord)
        #学習
        for i in range(iterations):  # 学習回数
            # - discriminatorの更新 -#
            for k in range(train_d):
                # - 実データの取得 -#
                images_real = sess.run(batch)
                # discriminatorの更新
                sess.run(train_step_D, feed_dict={_images_real: images_real})
            # - generatorの更新 -#
            sess.run(train_step_G)
            #tensorboardに書き込み
            w_summary = sess.run(summary, feed_dict={_images_real: images_real})
            writer.add_summary(w_summary, iterations)


            # 10000iteration毎に画像を保存
            if i % 100 == 0:
                #ディレクトリの作成
                file_name = "iterations:" + str(i) + ".png"
                save_name = save_image_dir +"/" + file_name
                res = sess.run(images_fake)#generatorの生成する画像を取得
                res = res.reshape(save_width * save_height , width, height)#channel情報を消すためにreshape
                res = np.uint8(np.clip(res,0.0,1.0)*255)#クリッピング後に、pillow用にuint8に変換
                data.visualize_image(res,save_width,save_height,mode="save",dir=save_name)
