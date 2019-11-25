import keras
from keras import Sequential

from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

class LeNet:
    """ Implementation of architecture LeNet-5 """

    @staticmethod
    def build(nRows, nCols, nChannels, nClasses, activation='relu'):
        """ Static methodd to create LeNet-5 model """

        # Criando um modelo sequencial
        model = keras.Sequential()

        # Adicionando uma camada convolucional, relu e max pooling
        model.add(Conv2D(20, 5, padding='same', input_shape=(nRows, nCols, nChannels)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Adicionando mais uma camada convolucional, relu e max pooling
        model.add(Conv2D(50, 5, padding='same'))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Adicionando nossa camada fully connected
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        # Adicionando a segunda camada fully connected (final)
        model.add(Dense(nClasses))
        model.add(Activation('softmax'))

        return model