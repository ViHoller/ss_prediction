from keras import regularizers
from keras.layers import (Layer, Convolution1D, Activation,
                          BatchNormalization, concatenate, Dropout)


class inception_conv(Layer):
    def __init__(self, kernel_s, num_features=100, **kwargs):
        super(inception_conv, self).__init__(**kwargs)
        self.conv = Convolution1D(num_features,
                                  kernel_size=kernel_s,
                                  kernel_regularizer=regularizers.l2(0.001),
                                  strides=1,
                                  padding='same')
        self.b_norm = BatchNormalization()

    def call(self, inputs):
        X = self.conv(inputs)
        X = Dropout(0.4)(X)
        X = self.b_norm(X)
        X = Activation('relu')(X)
        return X


class InceptionNet_paper(Layer):
    def __init__(self, **kwargs):
        super(InceptionNet_paper, self).__init__(**kwargs)
        self.conv1_1 = inception_conv(1)
        self.conv1_2 = inception_conv(1)
        self.conv1_3 = inception_conv(1)
        self.conv3_1 = inception_conv(3)
        self.conv3_2 = inception_conv(3)
        self.conv3_3 = inception_conv(3)
        self.conv3_4 = inception_conv(3)
        self.b_norm1 = BatchNormalization()
        self.b_norm2 = BatchNormalization()

    def call(self, inputs):
        X = self.b_norm1(inputs)
        X1 = self.conv1_1(X)
        X2 = self.conv3_1(self.conv1_2(X))
        X3 = self.conv3_4(self.conv3_3(self.conv3_2(self.conv1_3(X))))
        X = concatenate([X1, X2, X3])
        X = self.b_norm2(X)
        return X


class DeepInception_block(Layer):
    def __init__(self, **kwargs):
        super(DeepInception_block, self).__init__(**kwargs)
        self.inception1 = InceptionNet_paper()
        self.inception2_1 = InceptionNet_paper()
        self.inception2_2 = InceptionNet_paper()
        self.inception3_1 = InceptionNet_paper()
        self.inception3_2 = InceptionNet_paper()
        self.inception3_3 = InceptionNet_paper()
        self.inception3_4 = InceptionNet_paper()
        self.b_norm = BatchNormalization()

    def call(self, inputs):
        X1 = self.inception1(inputs)
        X2 = self.inception2_2(self.inception2_1(inputs))
        X3 = self.inception3_3(self.inception3_2(self.inception3_1(inputs)))
        X3 = self.inception3_4(X3)
        X = concatenate([X1, X2, X3])
        X = self.b_norm(X)
        return X
