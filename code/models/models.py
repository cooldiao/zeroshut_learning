import tensorflow as tf
import keras
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, Adagrad, Adamax
from keras.layers.core import Flatten
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers.core import Lambda
from keras import regularizers
from keras.regularizers import l2
from keras.layers import Dropout
import numpy as np
import os

def getLayer(input, size):
    a0 = Conv2D(int(size/2), (1, 1), padding='same')(input)
    a1 = Conv2D(int(size/2), (3, 3), padding='same')(input)
    a2 = Conv2D(int(size/2), (5, 5), padding='same')(input)
    b0 = Conv2D(size, (1, 1), padding='same')(a0)
    b1 = Conv2D(size, (5, 5), padding='same')(a1)
    b2 = Conv2D(size, (3, 3), padding='same')(a1)
    b3 = Conv2D(size, (5, 5), padding='same')(a2)
    b4 = Conv2D(size, (3, 3), padding='same')(a2)
    a = keras.layers.add([b0, b1, b2, b3, b4])
    mpl = AveragePooling2D(pool_size=2)(a)
    return BatchNormalization()(mpl)

def MyModel1(metrics='acc'):
	inputs = Input(shape=(64, 64, 3), dtype='float64')
	l1 = getLayer(inputs, 64)
	l2 = getLayer(l1, 128)
	# mpl = AveragePooling2D(pool_size=4)(l2)
	l3 = getLayer(l2, 256)
	l4 = getLayer(l3, 512)
	# x = AveragePooling2D(pool_size=4)(l4)
	flt = Flatten()(l4)
	# drop1 = Dropout(0.5)(flt)
	f1 = Dense(4096, activation='softplus')(flt)
	# drop2 = Dropout(0.5)(f1)
	f2 = Dense(30, activation='softsign')(f1)
	o_attr = Lambda(lambda x: (x+1)/2.)(f2)
	gap = GlobalAveragePooling2D()(l4)
	# drop3 = Dropout(0.5)(gap)
	f3 = Dense(30, activation='softplus')(gap)
	o_emb = Lambda(lambda x: x/2.)(f3)
	out = keras.layers.concatenate([o_attr, o_emb])
	# out = o_attr
	model = Model(inputs=inputs, outputs=out)
	# for layer in model.layers[0:47]:
	#    layer.trainable = False
	model.compile(optimizer=Adam(lr=0.00005), #lr=0.00005
              loss='categorical_crossentropy',
              metrics=[metrics])
	return model

def resnet_layer(inputs,
                 num_filters=32,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

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

def resnet_v1(depth, num_classes, metrics='acc'):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 32
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=(64, 64, 3))
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8, name='apl')(x)
    y = Flatten()(x)
    # outputs = Dense(num_classes,
    #                 activation='softplus',
    #                 kernel_initializer='he_normal')(y)
    outputs = Dense(285, activation='softmax', kernel_initializer='he_normal')(y)
    # outputs = Dense(num_classes,
    #                 activation='softmax',
    #                 kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=[metrics])
    return model

def zsl_res(metrics=['acc']):
    model = load_model(os.path.join(os.getcwd(), '../data/saved_models_zsl/zsl_model_res_pretrain.h5'))
    output = model.get_layer('apl').output
    flt = Flatten()(output)
    attr_out = Dense(6,
                    activation='softsign',
                    name='attr_out',
                    kernel_initializer='he_normal')(flt)
    emb_out = Dense(300,
                    activation='relu',
                    name='emb_out',
                    kernel_initializer='he_normal')(flt)
    model = Model(inputs=model.input, outputs=[attr_out, emb_out])
    for layer in model.layers[0:110]:
        layer.trainable = False
    model.compile(loss={"attr_out": 'poisson', "emb_out": "cosine_proximity"},
              optimizer=Adam(lr=lr_schedule(0)),
              loss_weights={'attr_out': 0.6, 'emb_out': 0.4},
              metrics=metrics)
    return model

def res_pretrain_finetune():
    model = load_model(os.path.join(os.getcwd(), '../data/saved_models_zsl/zsl_model_res_pretrain_finetune.h5'))
    return model

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-5
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def callbacks():
    save_dir = os.path.join(os.getcwd(), '../data/saved_models_zsl')
    model_name = 'zsl_model_res_pretrain_finetune2.{epoch:03d}.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='lose',
                             verbose=1,
                             save_best_only=False)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    return [checkpoint, lr_reducer, lr_scheduler]