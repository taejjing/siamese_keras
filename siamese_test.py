import time
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import shuffle

from util import gen_class_names, generate_one_hot_encoding
from model import get_siamese_model
from Siamese_pipeline import Siamese_Loader

from keras.optimizers import Adam
from keras import backend as K
K.set_image_data_format('channels_last')
#For 2D data (e.g. image), "channels_last" assumes (rows, cols, channels) while "channels_first" assumes (channels, rows, cols).
#For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while "channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).

# Check whether GPU is being or not
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

data_path = os.path.join("omniglot", "python")
train_folder = os.path.join(data_path, 'images_background')
valpath = os.path.join(data_path, 'images_evaluaton')

print(train_folder)
print(valpath)

# TODO: This code is just about checking an image file. Moving to anothoer file or deleting is better
# a_img_path = os.path.join(train_folder, 'Sanskrit/character11/0861_06.png')
# img = cv2.imread(a_img_path)
# print("Each image in the data set has a same of {0}".format(img.shape))
# flattened_img = img.flatten()
# print("The number of features in any image from the data set are: {0}".format(flattened_img.shape[0]))

base_class_name = 'character'
classes = gen_class_names(base_class_name) # util.py
labels = generate_one_hot_encoding(classes) # util.py

input_shape = (105, 105, 1)
model = get_siamese_model(input_shape) # model.py
model.summary()

optimizer = Adam(lr = 0.00006)
model.compile(loss="binary_crossentropy", optimizer=optimizer)

with open(os.path.join(data_path, "train.pickle"), "rb") as f:
    (X, classes) = pickle.load(f)

with open(os.path.join(data_path, "val.pickle"), "rb") as f:
    (Xval, val_classes) = pickle.load(f)
    
print("Training alphabets: \n")
print(list(classes.keys()))
print("Validation alphabets:", end="\n\n")
print(list(val_classes.keys()))

# Load Siamese class
loader = Siamese_Loader()
loader.load_data(data_path)

evaluate_every = 10 # interval for evaluating on one-shot tasks
loss_every = 20 # interval for printing loss ( iterations )
batch_size = 32
n_iter = 20000
N_way = 20 # how many classes for testing one-shot tasks
n_val = 250 # how many one-shot tasks to validate on?
best = -1

weights_f_name = "model_weights.h5"
weights_path = os.path.join("models", weights_f_name)

print("Starting training process !\n")
t_start = time.time()

for i in range(1, n_iter):
    ( inputs, targets ) = loader.get_batch(batch_size) # TODO: Why doesn't this code use 'generate' method?
    loss = model.train_on_batch(inputs, targets)
    print("\n------------------------------------------\n")
    print("Loss: {0}".format(loss))

    if i % evaluate_every == 0:
        print("Time for {0} iterations: {1}".format(i, time.time()-t_start))
        val_acc = loader.test_oneshot(model, N_way, n_val, verbose=True)
        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            print("Saving weights to : {0} \n".format(weights_f_name))
            model.save_weights(weights_path)
            best = val_acc
    if i % loss_every == 0:
        print("iteration {}, training loss: {:.2f},".format(i, loss))



