

import tensorflow as tf
from keras.layers import *

ACTIVATION = LeakyReLU

class Conv1DModule(tf.Module):

    def __init__(self, units, kernel_size = 3, strides=1, padding="same", batch_norm=True, name="Conv1DModule"):
        super(Conv1DModule, self).__init__(name=name)

        self.conv = Conv1D(units, kernel_size, strides=strides, padding=padding)
        self.bn = None
        if (batch_norm):
            self.bn = BatchNormalization()
        self.act = ACTIVATION()

    def __call__(self, x):
        x = self.conv(x)
        if (self.bn is not None):
            x = self.bn(x)
        x = self.act(x)
        return x
    
class Conv2DModule(tf.Module):

    def __init__(self, units, kernel_size = 3, strides=(1, 1), padding="same", batch_norm=True, name="Conv1DModule"):
        super(Conv2DModule, self).__init__(name=name)

        self.conv = Conv2D(units, kernel_size, strides=strides, padding=padding)
        self.bn = None
        if (batch_norm):
            self.bn = BatchNormalization()
        self.act = ACTIVATION()

    def __call__(self, x):
        x = self.conv(x)
        if (self.bn is not None):
            x = self.bn(x)
        x = self.act(x)
        return x
    

class DenseModule(tf.Module):
    
        def __init__(self, units, dropout=0.0, name="DenseModule"):
            super(DenseModule, self).__init__(name=name)
    
            self.dense = Dense(units)
            self.dropout = None
            if (dropout > 0):
                self.dropout = Dropout(dropout)
            self.act = ACTIVATION()
    
        def __call__(self, x):
            x = self.dense(x)
            if (self.dropout is not None):
                x = self.dropout(x)
            x = self.act(x)
            return x
        

class resLSTM(tf.Module):

    def __init__(self, units, layers=1, dropout=0.0, return_sequences=True, name="resLSTM"):
        super(resLSTM, self).__init__(name=name)

        self.lstms = []
        for i in range(layers):
            self.lstms.append(LSTM(units, return_sequences = (return_sequences or i != layers-1), dropout=dropout))
        self.return_sequences = return_sequences
        self.add = Add()
        self.conv = Conv1D(units, 1, padding="same", use_bias=False, activation="linear", kernel_initializer="ones")

    def __call__(self, x):
        if (self.return_sequences):
            lx = x
            for lstm in self.lstms:
                x = lstm(x)
            lx = self.conv(lx)
            x = self.add([x, lx])
        else:
            lx = x
            for lstm in self.lstms[:-1]:
                x = lstm(x)
            if (len(self.lstms) >= 2):
                lx = self.conv(lx)
                x = self.add([x, lx])
            x = self.lstms[-1](x)
        return x


nb_attention = 0
class Attention(tf.keras.layers.Layer):

    # take input as (BATCH_SIZE, TIME_STEPS, INPUT_DIM)
    # multiply each input with a weight along the time axis 
    # to measure the importance of each time step

    # W must be between 0 and 1

    # static instance number


    def __init__(self,heads=1, name="Attention"):
        global nb_attention
        nb_attention += 1
        super(Attention, self).__init__(name=name+"_"+str(nb_attention))
        self.heads = heads

        
    def build(self, input_shape):
        # MinMaxNorm
        self.w = self.add_weight(name="W_linear", 
                shape=(input_shape[1], 3), 
                trainable=True,
                initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1),
                constraint=tf.keras.constraints.NonNeg(),
                dtype=tf.float32)
    

    def call(self, x):
        y = []
        for i in range(self.heads):
            y.append(x * self.w[:,i:i+1])
        y = tf.concat(y, axis=-1)
        return y
    


class Inception1D_A(tf.Module):

    def __init__(self, filters, batch_norm = True, activation="relu", name="inception_A"):
        super(Inception1D_A, self).__init__(name=name)

        self.convA1 = Conv1D(filters, 1, padding="same", activation=None)

        self.convB1 = Conv1D(filters, 1, padding="same", activation=None)
        self.convB2 = Conv1D(filters, 3, padding="same", activation=None)

        self.convC1 = Conv1D(filters, 1, padding="same", activation=None)
        self.convC2 = Conv1D(int(filters * 1.5), 3, padding="same", activation=None)
        self.convC3 = Conv1D(int(filters * 2.0), 3, padding="same", activation=None)

        self.concat = Concatenate()
        self.merge_conv = Conv1D(filters, 1, padding="same", activation="linear")

        self.use_batch_norm = batch_norm
        if (self.use_batch_norm):
            self.batch_norm = BatchNormalization()

        self.skip_w = Conv1D(filters, 1, padding="same", activation="linear")
        self.add = Add()

        self.activation = Activation(activation)

    def __call__(self, x):
        l = x
        xA = self.convA1(x)

        xB = self.convB1(x)
        xB = self.convB2(xB)

        xC = self.convC1(x)
        xC = self.convC2(xC)
        xC = self.convC3(xC)

        x = self.concat([xA, xB, xC])
        x = self.merge_conv(x)

        if (self.use_batch_norm):
            x = self.batch_norm(x)

        if (l.shape[-1] != x.shape[-1]):
            l = self.skip_w(l)
        x = self.add([l, x])

        x = self.activation(x)
        return x

class Inception1D_B(tf.Module):

    def __init__(self, filters, batch_norm = True, activation="relu", name="inception_B"):
        super(Inception1D_B, self).__init__(name=name)

        self.convA1 = Conv1D(filters, 1, padding="same", activation=None)

        self.convB1 = Conv1D(filters, 1, padding="same", activation=None)
        self.convB2 = Conv1D(int(filters * 1.25), 7, padding="same", activation=None)
        self.convB3 = Conv1D(int(filters * 1.50), 7, padding="same", activation=None)

        self.concat = Concatenate()
        self.merge_conv = Conv1D(filters, 1, padding="same", activation="linear")

        self.use_batch_norm = batch_norm
        if (self.use_batch_norm):
            self.batch_norm = BatchNormalization()

            
        self.skip_w = Conv1D(filters, 1, padding="same", activation="linear")
        self.add = Add()

        self.activation = Activation(activation)

    def __call__(self, x):
        l = x
        xA = self.convA1(x)

        xB = self.convB1(x)
        xB = self.convB2(xB)
        xB = self.convB3(xB)

        x = self.concat([xA, xB])
        x = self.merge_conv(x)

        if (self.use_batch_norm):
            x = self.batch_norm(x)

        if (l.shape[-1] != x.shape[-1]):
            l = self.skip_w(l)
        x = self.add([l, x])

        x = self.activation(x)
        return x
    
class Inception1D_C(tf.Module):
    
        def __init__(self, filters, batch_norm = True, activation="relu", name="inception_C"):
            super(Inception1D_C, self).__init__(name=name)
    
            self.convA1 = Conv1D(filters, 1, padding="same", activation=None)

            self.convB1 = Conv1D(filters, 1, padding="same", activation=None)
            self.convB2 = Conv1D(int(filters * 1.1666), 3, padding="same", activation=None)
            self.convB3 = Conv1D(int(filters * 1.3333), 3, padding="same", activation=None)

            self.concat = Concatenate()
            self.merge_conv = Conv1D(filters, 1, padding="same", activation="linear")
    
            self.use_batch_norm = batch_norm
            if (self.use_batch_norm):
                self.batch_norm = BatchNormalization()
    
            self.skip_w = Conv1D(filters, 1, padding="same", activation="linear")
            self.add = Add()

    
            self.activation = Activation(activation)
    
        def __call__(self, x):
            l = x
            xA = self.convA1(x)

            xB = self.convB1(x)
            xB = self.convB2(xB)
            xB = self.convB3(xB)

            x = self.concat([xA, xB])
            x = self.merge_conv(x)

            if (self.use_batch_norm):
                x = self.batch_norm(x)

            if (l.shape[-1] != x.shape[-1]):
                l = self.skip_w(l)
            x = self.add([l, x])

            x = self.activation(x)
            return x
        
class Inception1D_RA(tf.Module):
    
        def __init__(self, filters, batch_norm = True, activation="relu", name="inception_RA"):
            super(Inception1D_RA, self).__init__(name=name)
    
            self.poolA1 = MaxPooling1D(3, strides=2, padding="valid")
            
            self.convB1 = Conv1D(int(filters * 1.5), 3, strides=2, padding="valid", activation=None)

            self.convC1 = Conv1D(filters, 1, padding="same", activation=None)
            self.convC2 = Conv1D(filters, 3, padding="same", activation=None)
            self.convC3 = Conv1D(int(filters * 1.5), 3, strides=2, padding="valid", activation=None)

            self.use_batch_norm = batch_norm
            if (self.use_batch_norm):
                self.batch_norm = BatchNormalization()

            self.concat = Concatenate()
            self.merge_conv = Conv1D(filters, 1, padding="same", activation="linear")


            self.activation = Activation(activation)

        def __call__(self, x):
            xA = self.poolA1(x)
            
            xB = self.convB1(x)
            
            xC = self.convC1(x)
            xC = self.convC2(xC)
            xC = self.convC3(xC)

            x = self.concat([xA, xB, xC])
            x = self.merge_conv(x)

            if (self.use_batch_norm):
                x = self.batch_norm(x)

            x = self.activation(x)
            return x
  
class Inception1D_RB(tf.Module):
    
        def __init__(self, filters, batch_norm = False, activation="relu", name="inception_RA"):
            super(Inception1D_RB, self).__init__(name=name)
    
            self.poolA1 = MaxPooling1D(3, strides=2, padding="valid")
            

            self.convB1 = Conv1D(filters, 1, padding="same", activation=None)
            self.convB2 = Conv1D(int(filters * 1.5), 3, strides=2, padding="valid", activation=None)

            self.convC1 = Conv1D(filters, 1, padding="same", activation=None)
            self.convC2 = Conv1D(int(filters * 1.125), 3, strides=2, padding="valid", activation=None)

            self.convD1 = Conv1D(filters, 1, padding="same", activation=None)
            self.convD2 = Conv1D(int(filters * 1.125), 3, padding="same", activation=None)
            self.convD3 = Conv1D(int(filters * 1.25), 3, strides=2, padding="valid", activation=None)

            self.use_batch_norm = batch_norm
            if (self.use_batch_norm):
                self.batch_norm = BatchNormalization()

            self.concat = Concatenate()
            self.merge_conv = Conv1D(filters, 1, padding="same", activation="linear")

            self.activation = Activation(activation)

        def __call__(self, x):
            xA = self.poolA1(x)
            
            xB = self.convB1(x)
            xB = self.convB2(xB)

            xC = self.convC1(x)
            xC = self.convC2(xC)

            xD = self.convD1(x)
            xD = self.convD2(xD)
            xD = self.convD3(xD)

            x = self.concat([xA, xB, xC, xD])
            x = self.merge_conv(x)

            if (self.use_batch_norm):
                x = self.batch_norm(x)

            x = self.activation(x)
            return x
        