from keras import regularizers
from keras.layers import (Layer, Convolution1D, Activation,
                          BatchNormalization, concatenate, Dropout,
                          Masking)


class inception_conv(Layer):
    def __init__(self, kernel_s, num_features=31, **kwargs):
        super(inception_conv, self).__init__(**kwargs)
        self.conv = Convolution1D(num_features,
                                  kernel_size=kernel_s,
                                  kernel_regularizer=regularizers.l2(0.001),
                                  strides=1,
                                  padding='same',
                                  activation='relu')
        self.b_norm = BatchNormalization()
        self.masking = Masking(mask_value=0)

    def call(self, inputs):

        X = self.conv(inputs)
        X = Dropout(0.4)(X)
        X = self.b_norm(X)
        return X


def InceptionNet(inputs):
    X = BatchNormalization()(inputs)
    X1 = inception_conv(1)(X)
    X2 = inception_conv(3)(inception_conv(1)(X))
    X3 = inception_conv(3)(inception_conv(3)(inception_conv(3)(inception_conv(1)(X))))
    X = concatenate([X1, X2, X3])
    X = BatchNormalization()(X)
    return X


def DeepInception_block(inputs):
    X1 = InceptionNet(inputs)
    X2 = InceptionNet(InceptionNet(inputs))
    X3 = InceptionNet(InceptionNet(InceptionNet(InceptionNet(inputs))))
    X = concatenate([X1, X2, X3])
    X = BatchNormalization()(X)
    return X
