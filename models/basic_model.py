from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Concatenate
from keras.applications.densenet import *
from numbers import Number
from keras.utils import to_categorical

from models.model import Model


class Basic(Model):
    """intput dimension is the shape of the input"""

    def __init__(self, input_dimension, output_dimension):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.model = Sequential()
        self.model.add(Flatten())
        self.model.add(Dense(300, input_shape=(512, 512)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(50))
        self.model.add(Activation('relu'))
        self.model.add(Dense(25))
        self.model.add(Activation('relu'))
        self.model.add(Dense(10))
        self.model.add(Activation('relu'))
        self.model.add(Dense(2))
        self.model.add(Activation('softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, X, y):
        self.model.fit(x=X, y=to_categorical(y), epochs=1, batch_size=8)

    def predict(self, X):
        self.model.predict(X)