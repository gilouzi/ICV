from keras.utils import np_utils
from keras.datasets import cifar10

class Cifar10:
    """ Class to read CIFAR-10 data """
    
    @staticmethod
    def read_data():
        """ Static method to read the data """

        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')
        train_images /= 255
        test_images /= 255
        
        # Criando os vetores de one-hot
        train_labels = np_utils.to_categorical(train_labels, 10)
        test_labels = np_utils.to_categorical(test_labels, 10)

        return (train_images, train_labels), (test_images, test_labels)