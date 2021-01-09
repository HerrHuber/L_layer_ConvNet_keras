# -*- coding: utf-8 -*-
# This Program

import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, Flatten, Dense
from keras.layers import MaxPooling2D
from keras.models import Model


def F1Score(pred, y):
    debug = 0
    if(debug):
        print(pred.shape)
        print(y.shape)
        start = 24
        end = 34
    epsilon = 0.0001  # to avoid division by 0
    if(debug):
        print("pred[0][start:end]: ", pred[0][start:end])
        print("y[0][start:end]: ", y[0][start:end])
    #TP = ((pred == 1) * pred) == ((y == 1) * y)  # (pred == y) == 1
    TP = np.logical_and(np.equal(pred, 1), np.equal(y, 1))
    if(debug):
        #print("(pred == 1): ", (pred == 1)[0][start:end])
        #print("(y == 1): ", (y == 1)[0][start:end])
        #print("(pred == 1) == (y == 1): ", (pred == 1) == (y == 1)[0][start:end])
        print("(pred == 1): ", np.equal(pred, 1)[0][start:end])
        print("(y == 1): ", np.equal(y, 1)[0][start:end])
        print("TP: ", TP[0][start:end])
        print()
    TP = np.sum(TP)
    if(TP == 0):
        TP += epsilon
    if(debug):
        print("TP: ", TP)

    #TN = (pred == 0) and (y == 0)  # (pred == y) == 0
    TN = np.logical_and(np.equal(pred, 0), np.equal(y, 0))
    if(debug):
        print("pred == 0: ", np.equal(pred, 0)[0][start:end])
        print("y == 0: ", np.equal(y, 0)[0][start:end])
        print("TN: ", TN[0][start:end])
        print()
    TN = np.sum(TN)
    if(debug):
        print("TN: ", TN)

    #FP = (pred == 1) and (y == 0)
    FP = np.logical_and(np.equal(pred, 1), np.equal(y, 0))
    if(debug):
        print("pred == 1: ", np.equal(pred, 1)[0][start:end])
        print("y == 0: ", np.equal(y, 0)[0][start:end])
        print("FP: ", FP[0][start:end])
        print()
    FP = np.sum(FP)
    if(FP == 0):
        FP += epsilon
    if(debug):
        print("FP: ", FP)

    #FN = (pred == 0) and (y == 1)
    FN = np.logical_and(np.equal(pred, 0), np.equal(y, 1))
    if(debug):
        print("pred == 0: ", np.equal(pred, 0)[0][start:end])
        print("y == 1: ", np.equal(y, 1)[0][start:end])
        print("FN: ", FN[0][start:end])
        print()
    FN = np.sum(FN)
    if(FN == 0):
        FN += epsilon
    if (debug):
        print("FN: ", FN)

    precision = TP / (TP + FP)
    if (debug):
        print("precision: ", precision)
    recall = TP / (TP + FN)
    if(debug):
        print("recall: ", recall)
    score = 2 / ((1 / precision) + (1 / recall))
    return precision, recall, score


def load_data(filename):
    """
    Dataset format should is h5 and look like this:
    datasets/
    --filename
    ----train_x (m, 64, 64, 3)
    ----train_y (m,)

    m := number of training examples
    filename := e.g. "mydataset.h5"
    """
    dataset = h5py.File(str(filename), "r")
    X = np.array(dataset["train_x"][:])
    Y = np.array(dataset["train_y"][:])
    # reshape from (m,) to (1, m)
    Y = Y.reshape((1, Y.shape[0]))

    return X, Y


def three_layer_ConvNet_keras(input_shape):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = Conv2D(16, (3, 3), strides=(1, 1), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model


def L_layer_ConvNet_keras(input_shape, layer_dims, strides, paddings, pool_filters, pool_strides, pool_paddings):
    """
    layer_dims = [[(8, 3, 3), (16, 3, 3)], [6, 1]]
    strides = [1, 1]
    paddings = ["same", "same"]
    pool_filters = [4, 4]
    pool_strides = [4, 4]
    pool_paddings = ["same", "same"]
    """
    conv_layers = layer_dims[0]
    dens_layers = layer_dims[1]
    L_conv = len(conv_layers)
    L_dens = len(dens_layers)

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    for l in range(L_conv):
        X = Conv2D(conv_layers[l][0], (conv_layers[l][1], conv_layers[l][2]), strides=(strides[l], strides[l]), padding=paddings[l], name='conv' + str(l))(X)
        X = BatchNormalization(axis=3, name='bn' + str(l))(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((pool_filters[l], pool_filters[0]), strides=(pool_strides[l], pool_strides[l]), padding=pool_paddings[l], name='max_pool' + str(l))(X)

    X = Flatten()(X)
    for l in range(L_dens-1):
        X = Dense(dens_layers[l])(X)
        X = Activation("relu")(X)

    X = Dense(dens_layers[-1], activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model


def main():
    print(time.time())

    filename = "../datasets/catvnoncat_3831_1315.h5"
    X_train, Y_train = load_data(filename)
    # X_train = preprocess(X_train)
    X_train = X_train / 255.
    Y_train = Y_train.T
    print("X_train.shape: ", X_train.shape)
    print("Y_train.shape: ", Y_train.shape)

    test_filename = "../datasets/train_catvnoncat.h5"
    test_dataset = h5py.File(test_filename, "r")
    X_test = np.array(test_dataset["train_set_x"][:])
    Y_test = np.array(test_dataset["train_set_y"][:])
    # reshape from (m,) to (1, m)
    Y_test = Y_test.reshape((1, Y_test.shape[0]))
    # X_test = preprocess(X_test)
    X_test = X_test / 255.
    Y_test = Y_test.T
    print("X_test.shape: ", X_test.shape)
    print("Y_test.shape: ", Y_test.shape)

    # (4, 4, 3, 8) = (filter height, filter width, channel, number of filters)
    layer_dims = [[(8, 3, 3), (16, 3, 3), (32, 3, 3), (64, 3, 3)], [512, 256, 128, 64, 32, 16, 8, 1]]
    strides = [1, 1, 1, 1]
    paddings = ["SAME", "SAME", "SAME", "SAME"]
    pool_filters = [2, 2, 2, 2]
    pool_strides = [2, 2, 2, 2]
    #              32,15, 7, 4
    pool_paddings = ["SAME", "SAME", "SAME", "SAME"]
    learning_rate = 0.0001
    num_epochs = 5
    minibatch_size = 64
    print_cost = True

    start = time.time()
    print("Start time: ", start)
    # execute model
    model = L_layer_ConvNet_keras(X_train.shape[1:], layer_dims, strides, paddings, pool_filters, pool_strides, pool_paddings)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x=X_train, y=Y_train, epochs=num_epochs, batch_size=minibatch_size)

    acc_loss_train = model.evaluate(x=X_train, y=Y_train)
    acc_loss_test = model.evaluate(x=X_test, y=Y_test)
    print()
    print("epoch " + str(num_epochs) + ":")
    print("Train Accuracy = " + str(acc_loss_train[1]))
    print("Test Accuracy = " + str(acc_loss_test[1]))

    diff = time.time() - start
    print("Time: ", diff)

    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    #print("preds_train[:10, 0]: ", preds_train[:10, 0])
    preds_train = (preds_train >= 0.5) * 1
    preds_test = (preds_test >= 0.5) * 1
    #print("preds_train[:10, 0]: ", preds_train[:10, 0])
    #print("Y_trains: ", Y_train[:10, 0])
    f1_train = F1Score(preds_train[:, 0], Y_train[:, 0])
    f1_test = F1Score(preds_test[:, 0], Y_test[:, 0])
    print()
    print("F1score train: ", f1_train[-1])
    print("F1score test: ", f1_test[-1])


if __name__ == "__main__":
    main()
#
#
#
#
#
#
#
#
#
#
