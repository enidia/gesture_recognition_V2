# -*- coding: utf-8 -*-
import cv2
import glob
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np
import time

# 模型保存地址
model_path='./models/106save03.h5'

# 读取图片
def read_img(path):
    cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        print('reading the images:%s' % (folder))
        for im in glob.glob(folder + '/*.jpg'):
            labels.append(idx)
            
            img = cv2.imread(im)
            img = cv2.resize(img,(100,100))
            
            mat = np.asarray(img) #image 转矩阵
            imgs.append(mat)

    return np.array(imgs), np.array(labels)


X_data , Y_data  = read_img('./storePic/')  # data 4038*(100,100,3)  label 4038个0~5

dict = np.unique(Y_data) #label 去重
print(dict)
# 打乱顺序
num_example = X_data.shape[0]  # 4038
arr = np.arange(num_example)  # [ 0 1 2 ... 4037]
np.random.shuffle(arr)  # 将arr乱序
X_data = X_data[arr]
Y_data = Y_data[arr]

# 将所有数据分为训练集和验证集
ratio = 0.8
s = np.int(num_example * ratio)
X_train = X_data[:s]
Y_train = Y_data[:s]
X_test  = X_data[s:]  # 验证集
Y_test  = Y_data[s:]

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(X_train[0].shape)

X_train = X_train / 255.0  # 归一化
X_test = X_test / 255.0

def create_model():
    model = keras.Sequential(
    [
        layers.Conv2D(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]),
                        filters=32, kernel_size=(5, 5), strides=(1,1), padding='same',
                       activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(200, activation='relu'),
        layers.Dense(11, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(),
             loss='sparse_categorical_crossentropy',
             # loss=keras.losses.SparseCategoricalCrossentropy(),
             metrics=['accuracy'])
    return model

deep_model = create_model()
deep_model.summary()

deep_model.fit(X_train, Y_train, batch_size=64, epochs=20)

deep_model.evaluate(X_test, Y_test)

deep_model.save(model_path)
