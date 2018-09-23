import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K


cate_labs = np.array([])
cate_attrs = np.array([])


def getClassAttr():
    attr = np.array(pd.read_csv('../data/attributes_per_class.txt', sep='\t', header=None))
    ret = {}
    for i in attr:
        ret[i[0]] = i[1:].astype(np.float64)
    return ret

def get_label_list():
    labels = np.array(pd.read_csv('../data/label_list.txt', sep='\t', header=None))
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

def getEmb():
    target,data = get_embedding()
    # pca = decomposition.PCA(n_components=30, whiten=True)
    # new_X = pca.fit_transform(data)
    emb = {}
    for i in range(len(target)):
        emb[target[i]] = data[i].astype(np.float32)
    return emb

def cate_data():
    global cate_labs, cate_attrs
    attr = np.array(pd.read_csv('../data/attributes_per_class.txt', sep='\t', header=None))
    _cate_attrs = np.array([]).reshape(0, 30).astype(np.float32)
    ret = {}
    for i in attr:
        cate_labs = np.append(cate_labs, i[0])
        _cate_attrs = np.row_stack((_cate_attrs, i[1:].astype(np.float32)))
    embs = getEmb()
    attr_out = np.array([])
    emb_out = np.array([])
    for i in range(len(_cate_attrs)):
        attr_out = np.append(attr_out, _cate_attrs[i][:6])
        emb_out = np.append(emb_out, embs[cate_labs[i]])
        # cate_attrs = np.append(cate_attrs, {"attr_out": _cate_attrs[i][:6], "emb_out": embs[cate_labs[i]]})
        # cate_attrs = np.row_stack((cate_attrs, _cate_attrs[i]))
    cate_attrs = {"attr_out": attr_out.reshape(-1, 6), "emb_out": emb_out.reshape(-1, 300)}
    return cate_attrs

cate_data()

size = 128

def repeat(x, dim, mul):
    x = tf.expand_dims(x,dim)
    return tf.tile(x, multiples = mul)

def d2(x, y):
    global size
    sub = tf.subtract(repeat(x['attr_out'], 1, [1, y['attr_out'].shape[0], 1]), repeat(y['attr_out'], 0, [int(size), 1, 1])) * 0.6 + tf.subtract(repeat(x['emb_out'], 1, [1, y['emb_out'].shape[0], 1]), repeat(y['emb_out'], 0, [int(size), 1, 1])) * 0.4
    sub = tf.multiply(sub, sub)
    return tf.reduce_sum(sub, axis=2)

def d2_attr(x, y):
    global size
    sub = tf.subtract(tf.cast(repeat(x, 1, [1, y.shape[0], 1]), tf.float32), tf.cast(repeat(y, 0, [int(size), 1, 1]), tf.float32))
    sub = tf.multiply(sub, sub)
    return tf.reduce_sum(sub, axis=2)

def d2_emb(x, y):
    global size
    sub = tf.subtract(tf.cast(repeat(x, 1, [1, y.shape[0], 1]), tf.float32), tf.cast(repeat(y, 0, [int(size), 1, 1]), tf.float32))
    sub = tf.multiply(sub, sub)
    return tf.reduce_sum(sub, axis=2)

def category(x, y):
    return tf.argmin(d2(x, y), 1)

def category_attr(x, y):
    return tf.argmin(d2_attr(x, y), 1)

def category_emb(x, y):
    return tf.argmin(d2_emb(x, y), 1)

def categorys(xs):
    global cate_attrs
    return category(xs, tf.Variable(cate_attrs))

def categorys_attr(xs):
    global cate_attrs
    return category_attr(xs, tf.Variable(cate_attrs['attr_out']))

def categorys_emb(xs):
    global cate_attrs
    return category_emb(xs, tf.Variable(cate_attrs['emb_out']))

def mean_pred(y_true, y_pred):
    pre = categorys(y_pred)
    y_true = categorys(y_true)
    # print((pre, y_true))
    # exit()
    return K.mean(tf.cast(tf.equal(pre, y_true), tf.float32))

def attr_acc(y_true, y_pred):
    pre = categorys_attr(y_pred)
    y_true = categorys_attr(y_true)
    # print((pre, y_true))
    # exit()
    return K.mean(tf.cast(tf.equal(pre, y_true), tf.float32))

def emb_acc(y_true, y_pred):
    pre = categorys_emb(y_pred)
    y_true = categorys_emb(y_true)
    return K.mean(tf.cast(tf.equal(pre, y_true), tf.float32))