import tensorflow as tf


class DyT(tf.keras.layers.Layer):
  def __init__(self, alpha_init_value=0.5):
    super(DyT, self).__init__()
    self.alpha_init_value = alpha_init_value

  def build(self, input_shape):
    self.alpha = self.add_weight("alpha",
                                 shape=[1],
                                 initializer=tf.constant_initializer(self.alpha_init_value)) 
    self.weight = self.add_weight("weight",
                                  shape=input_shape[1:],
                                  initializer=tf.ones_initializer())
    self.bias = self.add_weight("bias",
                                shape=input_shape[1:],
                                initializer=tf.zeros_initializer())

  def call(self, x):
    return self.weight * tf.tanh(self.alpha * x) + self.bias
    