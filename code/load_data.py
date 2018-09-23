import numpy as np
from skimage import io
import pandas as pd
import os


def getList():
    _list = np.array(pd.read_csv('../data/train.txt', sep='\t', header=None))
    ret = set()
    for i in _list:
        ret = ret.union(set([i[1]]))
    return ret

def getImg(file):
    img = io.imread('../data/DatasetB/train/'+file)
    # img = (img - 128) / 128.
    # img = img / 128. - .5
    return img

def getkeep():
    s = set()
    attrs = getClassAttr()
    attrs1 = getClassAttr()
    keep = set()
    for i in attrs:
        for j in attrs1:
            if i != j:
                d = np.sqrt(np.sum(np.square(attrs[i]-attrs[j])))
                if d == 0.:
                    keep = keep.union(set([i, j]))
    return list(keep)

test_count = 0
test_total = 36

def getDict():
    global test_count
    categorys = np.array(list(getList()))
    _attr = {}
    for i in range(30):
        _attr[i] = 0
    keeps = getkeep()
    attrs = getClassAttr()
    train = np.array([])
    test = np.array([])
    # 构建子属性包含分类个数
    for i in attrs:
        for j in range(len(attrs[i])):
            if attrs[i][j]>0 :
                _attr[j] += 1
    # 构建训练测试分类集
    for i in categorys:
        if i in keeps:
            train = np.append(train, i)
        else:
            if test_count >= test_total:
                train = np.append(train, i)
            else:
                flag = True
                for j in range(len(attrs[i])):
                    if (attrs[i][j]>0) and _attr[j]<31:
                        flag = False
                        break
                if flag:
                    test = np.append(test, i)
                    test_count += 1
                    for j in range(len(attrs[i])):
                        if attrs[i][j]>0:
                            _attr[j] -= 1
                else:
                    train = np.append(train, i)
    return (list(train), list(test))

def getClassAttr():
    attr = np.array(pd.read_csv('../data/attributes_per_class.txt', sep='\t', header=None))
    ret = {}
    for i in attr:
        ret[i[0]] = i[1:].astype(np.float64)
    return ret

def getLabel():
	ret = {}
	label = np.array(pd.read_csv('../data/label_list.txt', sep='\t', header=None))
	for i in range(len(label)):
			ret[label[i][0]] = i
	return ret

def load_data():
    labels = getLabel()
    # attr = getClassAttr()
    _list = np.array(pd.read_csv('../data/train.txt', sep='\t', header=None))
    np.random.shuffle(_list)
    np.random.shuffle(_list)
    _,test = getDict()
    x_train = np.array([]).reshape(0, 64, 64, 3)
    x_test = np.array([]).reshape(0, 64, 64, 3)
    y_train = np.array([]).reshape(0, 1)
    y_test = np.array([]).reshape(0, 1)
    save_dir = os.path.join(os.getcwd(), '../data/tmp/data')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    train_x = open("../data/tmp/data/train_x1","wb")
    train_y = open("../data/tmp/data/train_y1","wb")
    test_x = open("../data/tmp/data/test_x1","wb")
    test_y = open("../data/tmp/data/test_y1","wb")
    for i in _list:
        img = getImg(i[0]).astype(np.uint8)
        label = np.array([labels[i[1]]]).astype(np.uint8)[0]
        # label = np.zeros((len(labels))).astype(np.uint8)
        # label[labels[i[1]]] = 1
        if len(img.shape) != 3:
            continue
        if i[1] in test:
            test_x.write(img)
            test_y.write(label)
            # x_test = np.row_stack((x_test, [img]))
            # y_test = np.row_stack((y_test, label))
        else:
            train_x.write(img)
            train_y.write(label)
            # x_train = np.row_stack((x_train, [img]))
            # y_train = np.row_stack((y_train, label))
    #np.save('np_data/x_train.npy', x_train)
    #np.save('np_data/x_test.npy', x_test)
    #np.save('np_data/y_train.npy', y_train)
    #np.save('np_data/y_test.npy', y_test)
    # return (x_train, y_train, x_test, y_test)


#img = getImg('b20054b3508b7a3703257900bdd1a778.jpeg').astype(np.uint8)
#t = open('np_data/test_x', 'wb')
#t.write(img)
#t.write(img)
#exit()
load_data()