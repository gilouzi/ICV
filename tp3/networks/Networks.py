import keras
from keras import Sequential

from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Based on slides
class LeNet:
    """ Implementation of architecture LeNet-5 """

    @staticmethod
    def build(nRows, nCols, nChannels, nClasses, activation='relu'):
        """ Static method to create LeNet-5 model """

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

# Based on LeNet-5 architecture
class MyFirstCNN:
    """ Implementation of our own CNN architecture """

    @staticmethod
    def build(nRows, nCols, nChannels, nClasses, activation='relu'):
        """ Static method to create our own CNN model """

        # Criando um modelo sequencial
        model = keras.Sequential()

        # Adicionando uma camada convolucional, relu e max pooling
        model.add(Conv2D(16, 5, padding='same', input_shape=(nRows, nCols, nChannels)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Adicionando uma camada convolucional, relu e max pooling
        model.add(Conv2D(32, 5, padding='same'))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Adicionando mais uma camada convolucional, relu e max pooling
        model.add(Conv2D(64, 5, padding='same'))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Adicionando mais uma camada convolucional, relu e max pooling
        model.add(Conv2D(128, 5, padding='same'))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Adicionando nossa camada fully connected
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation(activation))

        # Adicionando a segunda camada fully connected (final)
        model.add(Dense(nClasses))
        model.add(Activation('softmax'))

        return model

# Based on slides
class MyFirstNN:
    """ Implementation of our own NN a architecture """

    @staticmethod
    def build(nRows, nCols, nChannels, nClasses, activation='relu'):
        """ Static method to create our own NN model """

        # Criando um modelo sequencial
        model = Sequential()

        # Adicionando uma hidden layer com 64 nodes
        model.add(Flatten())
        model.add(Dense(64, input_shape=(nRows, nCols, nChannels)))
        model.add(Activation(activation))

        # Adicionando uma hidden layer com 128 nodes
        model.add(Dense(128))
        model.add(Activation(activation))

        # Adicionando uma hidden layer com 64 nodes
        model.add(Dense(64))
        model.add(Activation(activation))

        # Adicionando um fully connected (final)
        model.add(Dense(nClasses))
        model.add(Activation('softmax'))

        return model

# Based on https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c
class MySecondCNN:
    """ Implementation of our own NN a architecture """

    @staticmethod
    def build(nRows, nCols, nChannels, nClasses, activation='relu'):
        """ Static method to create our own NN model """

        # Criando um modelo sequencial
        model = Sequential()

        # Adicionando uma camada convolucional, relu e max pooling
        model.add(Conv2D(16, 3, padding='same', input_shape=(nRows, nCols, nChannels)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Adicionando uma camada convolucional, relu e max pooling
        model.add(Conv2D(32, 3, padding='same'))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Adicionando mais uma camada convolucional, relu e max pooling
        model.add(Conv2D(64, 3, padding='same'))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Adicionando mais uma camada convolucional, relu e max pooling
        model.add(Conv2D(128, 3, padding='same'))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Adicionando as camadas fully connected
        model.add(Flatten())
        model.add(Dense(128, input_shape=(nRows, nCols, nChannels)))
        model.add(Activation(activation))

        model.add(Dense(256))
        model.add(Activation(activation))

        model.add(Dense(512))
        model.add(Activation(activation))

        # Adicionando um fully connected (final)
        model.add(Dense(nClasses))
        model.add(Activation('softmax'))

        return model

# Based on https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c
class MySecondNN:
    """ Implementation of our own NN a architecture """

    @staticmethod
    def build(nRows, nCols, nChannels, nClasses, activation='relu'):
        """ Static method to create our own NN model """

        # Criando um modelo sequencial
        model = Sequential()

        # Adicionando uma hidden layer com 128 nodes
        model.add(Flatten())
        model.add(Dense(128, input_shape=(nRows, nCols, nChannels)))
        model.add(Activation(activation))

        # Adicionando uma segunda hidden layer com 256 nodes
        model.add(Dense(256))
        model.add(Activation(activation))

        # Adicionando uma terceira hidden layer com 512 nodes
        model.add(Dense(512))
        model.add(Activation(activation))

        # Adicionando uma fully connected (final)
        model.add(Dense(nClasses))
        model.add(Activation('softmax'))

        return model