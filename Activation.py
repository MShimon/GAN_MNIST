import tensorflow as tf

#@brief:log関数　logの値が発散しないようにクリッピングしている
#@param:value_tf    Tensor
#@return:log(value_tf)
def log(value_tf):
    return tf.log(tf.clip_by_value(value_tf,1e-10,1.0))