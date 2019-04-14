import time
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from util import gen_class_names, generate_one_hot_encoding

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
from keras.regularizers import l2
K.set_image_data_format('channels_last')

from sklearn.utils import shuffle

# Check whether GPU is being or not
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

data_path = os.path.join("omniglot", "python")
train_folder = os.path.join(data_path, 'images_background')
valpath = os.path.join(data_path, 'images_evaluaton')

print(train_folder)
print(valpath)

a_img_path = os.path.join(train_folder, 'Sanskrit/character11/0861_06.png')
img = cv2.imread(a_img_path)
print("Each image in the data set has a same of {0}".format(img.shape))
flattened_img = img.flatten()
print("The number of features in any image from the data set are: {0}".format(flattened_img.shape[0]))


base_class_name = 'character'
classes = gen_class_names(base_class_name)
labels = generate_one_hot_encoding(classes)

def initialize_weights(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

# # Intialize bias with mean 0.0 and standard deviation of 10^-2
# weights = initialize_weights((1000,1))
# sns.distplot(weights)
# plt.title("Plot of weights initialized, with mean of 0.0 and standard deviation of 0.01")

def initialize_bias(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

# # Intialize bias with mean 0.5 and standard deviation of 10^-2
# bias = initialize_bias((1000,1))
# sns.distplot(bias)
# plt.title("Plot of biases initialized, with mean of 0.0 and standard deviation of 0.01")

def get_siamese_model(input_shape):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, \
            kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu', \
            kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', \
            kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', \
            kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid', \
            kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(1e-3)))
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    
    return siamese_net

input_shape = (105, 105, 1)
model = get_siamese_model(input_shape)
model.summary()

optimizer = Adam(lr = 0.00006)
model.compile(loss="binary_crossentropy",optimizer=optimizer)

with open(os.path.join(data_path, "train.pickle"), "rb") as f:
    (X, classes) = pickle.load(f)

with open(os.path.join(data_path, "val.pickle"), "rb") as f:
    (Xval, val_classes) = pickle.load(f)
    
print("Training alphabets: \n")
print(list(classes.keys()))
print("Validation alphabets:", end="\n\n")
print(list(val_classes.keys()))

