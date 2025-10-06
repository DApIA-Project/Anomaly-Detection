

import tensorflow as tf

class MulAttention(tf.keras.layers.Layer):
    
    def __init__(self):
        super(MulAttention, self).__init__()
        
    def build(self, input_shape):
        self.weight = self.add_weight("weight",
                                  shape=input_shape[1:],
                                  initializer=tf.ones_initializer())
        
    def call(self, x):
        return x * self.weight