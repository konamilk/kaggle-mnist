import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os
from pathlib import Path
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG

from keras import models
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, Input, LeakyReLU, Dense, Activation, Flatten, Dropout, MaxPool2D

VERSION = '001'  # 実験番号


def create_logger(exp_version):
    log_file = ('{}.log'.format(exp_version))

    # logger
    logger_ = getLogger(exp_version)
    logger_.setLevel(DEBUG)

    # formatter
    fmr = Formatter('[%(levelname)s] %(asctime)s >>\t%(message)s')

    # file handler
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    fh.setFormatter(fmr)

    # stream handler
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(fmr)

    logger_.addHandler(fh)
    logger_.addHandler(ch)


def get_logger(exp_version):
    return getLogger(exp_version)


if __name__ == '__main__':
    create_logger(VERSION)
    logger = get_logger(VERSION)
    logger.info('start')

    logger.info('Data Preparation')
    df = pd.read_csv('../input/digit-recognizer/train.csv')

    X_all = df.drop(['label'], axis=1).values.astype('float32')
    y_all = df['label'].values.astype('int32')

    sample_size = X_all.shape[0]
    train_size = int(sample_size * 0.9)
    validation_size = sample_size - train_size

    X_train = X_all[:train_size, :].reshape([train_size, 28, 28, 1])
    y_train = y_all[:train_size].reshape([train_size, 1])

    X_val = X_all[train_size:, :].reshape([validation_size, 28, 28, 1])
    y_val = y_all[train_size:].reshape([validation_size, 1])

    df_test = pd.read_csv('../input/digit-recognizer/test.csv')
    X_test = np.asarray(df_test.iloc[:, :]).reshape([-1, 28, 28, 1])

    X_train = X_train / 255
    X_val = X_val / 255
    X_test = X_test / 255

    logger.info('Model Definition')

    model = models.Sequential()

    model.add(Conv2D(32, 3, padding='same', input_shape=(28, 28, 1)))
    model.add(LeakyReLU())
    model.add(Conv2D(32, 3, padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, 3, padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(64, 3, padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))

    initial_lr = 0.001
    loss = 'sparse_categorical_crossentropy'
    model.compile(Adam(lr=initial_lr), loss=loss, metrics=['accuracy'])
    logger.info(model.summary())

    logger.info('#Training begin')

    epochs = 20
    batch_size = 256
    history_1 = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                          validation_data=(X_val, y_val))

    logger.info('#Training end')

    # plot accuracy and loss curve
    f = plt.figure(figsize=(20, 7))

    f.add_subplot(121)

    plt.plot(history_1.epoch, history_1.history['accuracy'], label='accuracy')
    plt.plot(history_1.epoch, history_1.history['val_accuracy'], label='val_accuracy')

    plt.title('Accuracy Curve', fontsize=18)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)

    f.add_subplot(122)

    plt.plot(history_1.epoch, history_1.history['loss'], label='loss')
    plt.plot(history_1.epoch, history_1.history['val_loss'], label='val_loss')

    plt.title('Loss Curve', fontsize=18)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.show()

    # Confusion Matrix
    p_val = np.argmax(model.predict(X_val), axis=1)

    error = 0
    confusion_matrix = np.zeros([10, 10])

    for i in range(X_val.shape[0]):
        confusion_matrix[y_val[i], p_val[i]] += 1
        if y_val[i] != p_val[i]:
            error += 1

    logger.info('Confusion Matrix : ')
    logger.info('{}'.format(confusion_matrix))
    logger.info('Errors in validation set : {}'.format(error))
    logger.info('Error Percentage : {}'.format((error * 100) / p_val.shape[0]))
    logger.info('Accuracy :{}'.format(100 - (error * 100) / p_val.shape[0]))
    logger.info('Validation set Shape : {}'.format(p_val.shape[0]))

    # plot Confusion Matrix
    f = plt.figure(figsize=(10, 8.5))
    f.add_subplot(111)

    plt.imshow(np.log2(confusion_matrix + 1), cmap='Reds')
    plt.colorbar()
    plt.tick_params(size=6, color='white')
    plt.xticks(np.arange(0, 10), np.arange(0, 10))
    plt.yticks(np.arange(0, 10), np.arange(0, 10))

    threshold = confusion_matrix.max() / 2

    for i in range(10):
        for j in range(10):
            plt.text(j, i, int(confusion_matrix[i, j]), horizontalalignment='center',
                     color='white' if confusion_matrix[i, j] > threshold else 'black')

    plt.xlabel('predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Create submission
    p_test = np.argmax(model.predict(X_test), axis=1)

    df_submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
    df_submission.loc[:, 'Label'] = p_test

    df_submission.to_csv('submission.csv', index=False)

    get_logger(VERSION).info('end')
