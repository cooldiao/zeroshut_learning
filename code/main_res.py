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
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import pandas as pd
import os
from models.models import resnet_v1
from models.models import zsl_res
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

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

imgs = np.array([])
labs = np.array([]).reshape(0, 30)

def parseFile(file, ret, div = "\t"):
    with open(file) as file_object:
        contents = file_object.read()
        contents = contents.split("\n")
        for content in contents:
            if(content != ''):
                content = content.split(div)
                ret = np.row_stack((ret, content))
    return ret

def getClassAttr():
    attr = np.array(pd.read_csv('../data/attributes_per_class.txt', sep='\t', header=None))
    ret = {}
    for i in attr:
        ret[i[0]] = i[1:].astype(np.float64)
    return ret

cate_labs = np.array([])
cate_attrs = np.array([]).reshape(0, 60).astype(np.float32)

def get_label_list():
    labels = parseFile("../data/label_list.txt", np.array([]).reshape(0, 2))
    _list = {}
    for i in labels:
        _list[i[0]] = i[1]
    return _list

def get_embedding():
    embeddings = np.array(pd.read_csv('../data/class_wordembeddings.txt', sep=' ', header=None))
    _list = {}
    for i in embeddings:
        _list[i[0]] = i[1:].astype(np.float64)
    labels = get_label_list()
    data = np.array([]).reshape(0, 300)
    target = np.array([]).reshape(0)
    for i in labels:
        target = np.append(target, i)
        data = np.row_stack((data, _list[labels[i]].reshape(300)))
    return (target, data)

def cate_data():
    global cate_labs, cate_attrs
    attr = np.array(pd.read_csv('../data/attributes_per_class.txt', sep='\t', header=None))
    _cate_attrs = np.array([]).reshape(0, 30).astype(np.float32)
    ret = {}
    for i in attr:
        cate_labs = np.append(cate_labs, i[0])
        _cate_attrs = np.row_stack((_cate_attrs, i[1:].astype(np.float32)))
    embs = getEmb()
    for i in range(len(_cate_attrs)):
        cate_attrs = np.row_stack((cate_attrs, np.append(_cate_attrs[i], embs[cate_labs[i]])))
        # cate_attrs = np.row_stack((cate_attrs, _cate_attrs[i]))

def getLabel():
    ret = {}
    label = np.array(pd.read_csv('../data/label_list.txt', sep='\t', header=None))
    for i in range(len(label)):
            ret[label[i][0]] = i
    return ret

def getEmb1():
    target,data = get_embedding()
    labels = getLabel()
    pca = decomposition.PCA(n_components=30, whiten=True)
    new_X = pca.fit_transform(data)
    emb = {}
    for i in range(len(target)):
        emb[labels[target[i]]] = new_X[i]
    return emb

def getEmb():
    target,data = get_embedding()
    pca = decomposition.PCA(n_components=30, whiten=True)
    new_X = pca.fit_transform(data)
    emb = {}
    for i in range(len(target)):
        emb[target[i]] = new_X[i].astype(np.float32)
    return emb

def parseLabel():
    global y_train,y_test,num_classes
    y_train_tmp = np.array([]).reshape(0, num_classes)
    y_test_tmp = np.array([]).reshape(0, num_classes)
    attr = np.array(pd.read_csv('../data/attributes_per_class.txt', sep='\t', header=None))
    _cate_attrs = np.array([]).reshape(0, 30).astype(np.float32)
    ret = {}
    for i in attr:
        _cate_attrs = np.row_stack((_cate_attrs, i[1:].astype(np.float32)))
    embs = getEmb1()
    for i in range(len(y_train)):
        y_train_tmp = np.row_stack((y_train_tmp, np.append(_cate_attrs[y_train[i]], embs[y_train[i]])))
    for i in range(len(y_test)):
        y_test_tmp = np.row_stack((y_test_tmp, np.append(_cate_attrs[y_test[i]], embs[y_test[i]])))
    y_train = y_train_tmp
    y_test = y_test_tmp
    # train = open("np_data/train_y_label"+str(num_classes),"wb")
    # test = open("np_data/test_y_label"+str(num_classes),"wb")
    # train.write(y_train.astype(np.uint8))
    # test.write(y_test.astype(np.uint8))

# cate_data()
# parseLabel()

def repeat(x, dim, mul):
    x = tf.expand_dims(x,dim)
    return tf.tile(x, multiples = mul)

def d2(x, y):
    global size
    sub = tf.subtract(repeat(x, 1, [1, y.shape[0], 1]), repeat(y, 0, [int(size), 1, 1]))
    sub = tf.multiply(sub, sub)
    return tf.reduce_sum(sub, axis=2)

def category(x, y):
    return tf.argmin(d2(x, y), 1)

def categorys(xs):
    global cate_attrs
    return category(xs, tf.Variable(cate_attrs))

def mean_pred(y_true, y_pred):
    pre = categorys(y_pred)
    y_true = categorys(y_true)
    # print((pre, y_true))
    # exit()
    return K.mean(tf.cast(tf.equal(pre, y_true), tf.float32))

model = resnet_v1(depth=32, num_classes=num_classes, metrics='acc')
# model = zsl_res(mean_pred)
# exit()


# model.load_weights("weights/my1/w2.h5", by_name=False)
# model = load_model("weights/my1/w1.h5")

save_dir = os.path.join(os.getcwd(), '../data/saved_models_zsl')
model_name = 'zsl_model_res_pretrain.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_mean_pred',
                             verbose=1,
                             save_best_only=False)

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 5e-5
    if epoch > 180:
        lr *= 0.5e-6
    elif epoch > 160:
        lr *= 1e-6
    elif epoch > 120:
        lr *= 5e-6
    elif epoch > 80:
        lr *= 1e-5
    print('Learning rate: ', lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

model.fit(x_train, y_train,
      batch_size=size,
      epochs=epochs,
      validation_data=(x_test, y_test),
      shuffle=True,
      callbacks=callbacks)

model.save("../data/saved_models_zsl/zsl_model_res_pretrain.h5")
exit()

def tag(out, embs):
    _tag = 0
    _min = 9999999999
    for i in embs:
        mse = np.sum(np.square(out - embs[i]))
        if(_min >= mse):
            _tag = i
            _min = mse
    return _tag

def predict(image, embs):
    global x_train_mean
    img = io.imread('../data/DatasetB/test/'+image)/255.
    if(len(img.shape)<3) or img.shape[2] < 3:
        img = np.repeat(img.reshape(64,64,1), 3, axis=2)
    img -= x_train_mean
    out = model.predict([[img]])
    return tag(out, embs)

def submit():
    images = np.array(pd.read_csv('../data/image.txt', sep='\t', header=None))
    submit_file = open('../submit/submit.txt', 'w')
    embs = getEmb1()
    for i in images:
        _predict = predict(i[0], embs)
        # print(_predict)
        # exit()
        submit_file.write(i[0])
        submit_file.write("\t")
        submit_file.write(_predict)
        submit_file.write("\n")

submit()


# # Score trained model.
# scores = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
