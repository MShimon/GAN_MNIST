import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import conv2d,conv2d_transpose,flatten
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import l2_regularizer
#活性化関数
relu = tf.nn.relu
sigmoid = tf.nn.sigmoid
#自作module
from Activation import log

#generatorクラス
class Discriminator():
    #@brief:コンストラクタ
    #@param:lr  学習率
    #       weight_decay    l2正則化の係数（現在は未使用）
    def __init__(self,lr,weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay
        self.scope = "discriminator"

    #@brief:順伝播計算を行うメソッド
    #@param:img   realdata or generatorの生成画像
    #@return:順伝播の結果
    def forward(self,img):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            net = conv2d(img, 64, [4, 4], stride=2, activation_fn=relu, normalizer_fn=batch_norm, weights_regularizer=l2_regularizer(self.weight_decay))
            net = conv2d(net, 128, [4, 4], stride=2, activation_fn=relu, normalizer_fn=batch_norm, weights_regularizer=l2_regularizer(self.weight_decay))
            net = flatten(net)
            net = fully_connected(net, 1024, activation_fn=relu, weights_regularizer=l2_regularizer(self.weight_decay))
            net = fully_connected(net, 1, activation_fn=sigmoid, weights_regularizer=l2_regularizer(self.weight_decay))
        return net

    #@brief:discriminatorのloss
    #@param:output_D_fake   ノイズを入力したときのdiscriminatorの出力
    #       output_D_real   real_dataを入力したときのdiscriminatorの出力
    #@return:lossの値
    def loss(self,output_D_fake,output_D_real):
        return -(tf.reduce_mean(log(output_D_real)) + tf.reduce_mean(log(1 - output_D_fake)))

    #@brief:discriminatorの学習を行う
    #@param:loss   誤差の値
    #@return:train_step
    def train(self,loss):
        # discriminatorのみを訓練
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        return tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(loss,var_list=train_vars)
