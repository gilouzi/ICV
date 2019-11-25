import keras
import argparse

from datetime import datetime

from networks.net import LeNet

from keras.utils import np_utils
from keras.optimizers import SGD
from keras.datasets import cifar10

if __name__ == '__main__':
    # Criando argumentos para o nosso código
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.01, help='Parâmetro que define o learning rate', type=float)
    parser.add_argument('--batch_size', default=128, help='Parâmetro que define o tamanho do batch', type=int)
    parser.add_argument('--epochs', default=10, help='Parâmetro que define o número de épocas', type=int)

    args = parser.parse_args()

    # Definindo o número de classes e tamanho de cada imagem
    nClasses = 10
    imgWidth = 32
    imgHeight = 32

    # Definindo o diretório de log para o TensorBoard
    logdir = 'logs/LeNet' + datetime.now().strftime('%Y/%m/%d-%H:%M:%S')
    tensorboard = keras.callbacks.TensorBoard(log_dir=logdir)

    # Lendo o dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    train_images /= 255
    test_images /= 255
    
    # Criando os vetores de hot-ones
    train_labels = np_utils.to_categorical(train_labels, nClasses)
    test_labels = np_utils.to_categorical(test_labels, nClasses)

    # Criando o modelo LeNet-5
    model = LeNet.build(imgWidth, imgHeight, 3, nClasses)

    # Compilando o modelo
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=args.lr), metrics=['accuracy'])

    # Treinando o modelo
    model.fit(train_images, train_labels, epochs=args.epochs, batch_size=args.batch_size, callbacks=[tensorboard])

    # Testando com os dados de teste
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)

    # Printando os resultados
    print('\nTest accuracy:', test_acc)
    print('Test loss:', test_loss)