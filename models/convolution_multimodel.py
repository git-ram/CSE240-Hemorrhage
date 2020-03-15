from keras.models import Sequential
from keras.layers import Dense, Conv2D, LeakyReLU, MaxPooling2D, Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Concatenate
from keras.applications.densenet import *
from numbers import Number
from keras.utils import to_categorical
import keras

from models.model import Model
import numpy as np
from keras import backend as K


class CNNMultilabel(Model):
    """intput dimension is the shape of the input"""

    def __init__(self, img_rows, img_cols):
        self.img_rows = img_rows
        self.img_cols = img_cols
        if  K.image_data_format() == 'channels_first':

            self.input_shape = (1,img_rows,img_cols)
        else:
            self.input_shape = (img_rows,img_cols,1)

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(3, 3)))
        self.model.add(Conv2D(64, kernel_size=(3, 3),
                              ))
        self.model.add(MaxPooling2D(pool_size=(3, 3)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(5, activation='sigmoid'))

        self.model.compile(loss=keras.losses.binary_crossentropy(),
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

    def fit(self, X, y):

        if K.image_data_format() == 'channels_first':
            X = X.reshape(X.shape[0], 1, self.img_rows, self.img_cols)
        else:
            X =  X.reshape( X.shape[0], self.img_rows, self.img_cols, 1)
        self.model.fit(x=X, y=y, epochs=1, batch_size=8)

    def predict(self, X):
        if K.image_data_format() == 'channels_first':
            X = X.reshape(X.shape[0], 1, self.img_rows, self.img_cols)
        else:
            X = X.reshape(X.shape[0], self.img_rows, self.img_cols, 1)
        pred = self.model.predict(X)
        return pred