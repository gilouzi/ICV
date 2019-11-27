import keras

from datetime import datetime

from keras.optimizers import SGD

from networks.Networks import LeNet
from networks.DataLoader import Cifar10

if __name__ == '__main__':
    # Lendo os dados
    (train_images, train_labels), (test_images, test_labels) = Cifar10.read_data()

    # Definindo as dimensões das imagens
    imgWidth = 32
    imgHeight = 32
    nClasses = 10

    # Definindo os possíveis learning rates, epochs e batch_sizes
    lr_list = [0.1, 0.01, 0.001]
    epochs_list = [10, 20, 30]
    batch_list = [64, 128, 256]

    best_loss = 0.0
    best_acc = 0.0

    for lr in lr_list:
        for epoch in epochs_list:
            for batch_size in batch_list:
                model = LeNet.build(imgHeight, imgWidth, 3, nClasses)
                opt = SGD(lr=lr)
                model.compile(optimizer=opt,
                              loss='categorical_crossentropy', 
                              metrics=['accuracy'])
                model.fit(train_images, train_labels,
                          epochs=epoch,
                          batch_size=batch_size,
                          validation_data=(test_images, test_labels))
                test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
                if test_acc > best_acc:
                    best_loss = test_loss
                    best_acc = test_acc
                    best_params = [lr, epoch, batch_size]

    print('Best loss:', best_loss)
    print('Best acc:', best_acc)
    print('Best hyperparams:', best_params)