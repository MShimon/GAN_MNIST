import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import conv2d,conv2d_transpose,flatten
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import l2_regularizer
#活性化関数
lrelu = tf.nn.leaky_relu
sigmoid = tf.nn.sigmoid
#自作module
from Activation import log

#generatorクラス
class Generator():
    #@brief:コンストラクタ
    #@param:lr  学習率
    #       weight_decay    l2正則化の係数（現在は未使用）
    def __init__(self,lr,weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay
        self.scope = "generator"

    #@brief:順伝播計算を行うメソッド
    #@param:noise   ノイズ
    #@return:順伝播の結果
    def forward(self,noise):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            net = fully_connected(noise, 1024, activation_fn=lrelu, weights_regularizer=l2_regularizer(self.weight_decay))
            net = fully_connected(net, 7 * 7 * 256, activation_fn=lrelu, weights_regularizer=l2_regularizer(self.weight_decay))
            net = tf.reshape(net, [-1, 7, 7, 256])
            net = conv2d_transpose(net, 64, [4, 4], stride=2, activation_fn=lrelu, normalizer_fn=batch_norm,
                                   weights_regularizer=l2_regularizer(self.weight_decay))
            net = conv2d_transpose(net, 32, [4, 4], stride=2, activation_fn=lrelu, normalizer_fn=batch_norm,
                                   weights_regularizer=l2_regularizer(self.weight_decay))
            net = conv2d(net, 1, 4, normalizer_fn=None, activation_fn=tf.tanh, weights_regularizer=l2_regularizer(self.weight_decay))

        return net

    #@brief:genenratorのloss
    #@param:output_D_fake   ノイズを入力したときのdiscriminatorの出力
    #@return:lossの値
    def loss(self,output_D_fake):
        return -tf.reduce_mean(log(output_D_fake))

    #@brief:genenratorの学習を行う
    #@param:loss   誤差の値
    #@return:train_step
    def train(self,loss):
        #generatorのみを訓練
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        return tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(loss,var_list=train_vars)