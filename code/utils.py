import numpy as np
import pandas as pd
import os
from skimage import io
from metrics import cate_data

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

def getLabel():
    ret = {}
    label = np.array(pd.read_csv('../data/label_list.txt', sep='\t', header=None))
    for i in range(len(label)):
            ret[label[i][0]] = i
    return ret

def getEmb1():
    target,data = get_embedding()
    labels = getLabel()
    # pca = decomposition.PCA(n_components=30, whiten=True)
    # new_X = pca.fit_transform(data)
    emb = {}
    for i in range(len(target)):
        emb[labels[target[i]]] = data[i].astype(np.float32)
    return emb

def parseLabel(y_train,y_test):
    return ({"attr_out": np.load("../data/tmp/data/y_train_attr.npy"), "emb_out": np.load("../data/tmp/data/y_train_emb.npy")}, {"attr_out": np.load("../data/tmp/data/y_test_attr.npy"), "emb_out": np.load("../data/tmp/data/y_test_emb.npy")})
    y_train_tmp = np.array([])
    y_test_tmp = np.array([])
    attr = np.array(pd.read_csv('../data/attributes_per_class.txt', sep='\t', header=None))
    _cate_attrs = np.array([]).reshape(0, 30).astype(np.float32)
    ret = {}
    for i in attr:
        _cate_attrs = np.row_stack((_cate_attrs, i[1:].astype(np.float32)))
    embs = getEmb1()
    attr_out = np.array([])
    emb_out = np.array([])
    for i in range(len(y_train)):
        attr_out = np.append(attr_out, _cate_attrs[y_train[i]][:6])
        emb_out = np.append(emb_out, embs[y_train[i]])
        # y_train_tmp = np.append(y_train_tmp, {"attr_out": _cate_attrs[y_train[i]][:6], "emb_out": embs[y_train[i]]})
    y_train_tmp = {"attr_out": attr_out.reshape(-1, 6), "emb_out": emb_out.reshape(-1, 300)}
    attr_out = np.array([])
    emb_out = np.array([])
    for i in range(len(y_test)):
        attr_out = np.append(attr_out, _cate_attrs[y_test[i]][:6])
        emb_out = np.append(emb_out, embs[y_test[i]])
        # y_test_tmp = np.append(y_test_tmp, {"attr_out": _cate_attrs[y_test[i]][:6], "emb_out": embs[y_test[i]]})
    y_test_tmp = {"attr_out": attr_out.reshape(-1, 6), "emb_out": emb_out.reshape(-1, 300)}
    np.save("../data/tmp/data/y_train_attr", y_train_tmp['attr_out'])
    np.save("../data/tmp/data/y_train_emb", y_train_tmp['emb_out'])
    np.save("../data/tmp/data/y_test_attr", y_test_tmp['attr_out'])
    np.save("../data/tmp/data/y_test_emb", y_test_tmp['emb_out'])
    return (y_train_tmp, y_test_tmp)
    # train = open("np_data/train_y_label"+str(num_classes),"wb")
    # test = open("np_data/test_y_label"+str(num_classes),"wb")
    # train.write(y_train.astype(np.uint8))
    # test.write(y_test.astype(np.uint8))

def tag(out, cate_datas, labels):
    _tag = 0
    _min = 9999999999
    for i in range(len(cate_datas['attr_out'])):
        mse = np.sqrt(np.sum(np.square(out[0] - cate_datas['attr_out'][i]))/len(out[0])) * 0.6 + np.sqrt(np.sum(np.square(out[1] - cate_datas['emb_out'][i]))/len(out[1])) * 0.4
        if(_min >= mse):
            _tag = i
            _min = mse
    return labels[_tag]

def predict(model, x_train_mean, image, cate_datas, labels):
    img = io.imread('../data/DatasetB/test/'+image)/255.
    if(len(img.shape)<3) or img.shape[2] < 3:
        img = np.repeat(img.reshape(64,64,1), 3, axis=2)
    img -= x_train_mean
    out = model.predict([[img]])
    return tag(out, cate_datas, labels)

def submit(model, x_train_mean):
    images = np.array(pd.read_csv('../data/image.txt', sep='\t', header=None))
    submit_file = open('../submit/submit3.txt', 'w')
    cate_datas = cate_data()
    labels = getLabel()
    labels = dict(zip(labels.values(), labels.keys()))
    for i in images:
        _predict = predict(model, x_train_mean, i[0], cate_datas, labels)
        submit_file.write(i[0])
        submit_file.write("\t")
        submit_file.write(_predict)
        submit_file.write("\n")