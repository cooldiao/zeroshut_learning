import numpy as np
from skimage import io
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, Adagrad, Adamax
from keras.layers.core import Flatten
from keras.models import load_model
from matplotlib import pyplot as plot
from keras.optimizers import RMSprop
from keras.optimizers import Nadam
from keras.layers import Dense, GlobalAveragePooling2D
import time
from keras.layers.core import Lambda
from sklearn import decomposition
from keras import regularizers
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import pandas as pd
import os
from models.models import resnet_v1, zsl_res, callbacks, res_pretrain_finetune
from metrics import mean_pred, attr_acc, emb_acc
from utils import parseLabel, submit
# from load_data import load_data

sess = tf.Session()
K.set_session(sess)
size = 128
epochs=50
num_classes = 285

# x_train, y_train, x_test, y_test = load_data()
x_train = np.fromfile("../data/tmp/data/train_x1", dtype=np.uint8).reshape(-1, 64, 64, 3)
x_test = np.fromfile("../data/tmp/data/test_x1", dtype=np.uint8).reshape(-1, 64, 64, 3)
y_train = np.fromfile("../data/tmp/data/train_y1", dtype=np.uint8)
y_test = np.fromfile("../data/tmp/data/test_y1", dtype=np.uint8)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


x_train = x_train[:size*int(len(x_train)/size)]
y_train = y_train[:size*int(len(y_train)/size)]
x_test = x_test[:size*int(len(x_test)/size)]
y_test = y_test[:size*int(len(y_test)/size)]
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean
print("start parseLabel")
y_train,y_test = parseLabel(y_train,y_test)
print("end parseLabel")

# model = resnet_v1(depth=32, num_classes=num_classes, metrics='acc')
model = zsl_res(metrics = {"attr_out": attr_acc, "emb_out": emb_acc})
# model = res_pretrain_finetune()
model.summary()
model.load_weights("../data/saved_models_zsl/zsl_model_res_pretrain_finetune.050.h5", by_name=False)
# model = load_model("weights/my1/w1.h5")
model.fit(x_train, y_train,
      batch_size=size,
      epochs=epochs,
      validation_data=(x_test, y_test),
      shuffle=True,
      callbacks=callbacks())
model.save("../data/saved_models_zsl/zsl_model_res_pretrain_finetune2.h5")
submit(model, x_train_mean)
