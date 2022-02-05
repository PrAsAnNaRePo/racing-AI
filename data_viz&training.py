import random
from collections import Counter
import keras
import keras.layers
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow.keras.losses
from tensorflow.keras.applications import xception, resnet_v2, inception_v3

from tensorflow.keras import layers, optimizers

c = 0
DATSET_PATH = 'dataset'
data = []
for i in os.listdir(f'{DATSET_PATH}/'):
    d = np.load(f'{DATSET_PATH}/{i}', allow_pickle=True)
    data.extend(d)
    c += 1
    # print(f'dataset {c} loaded')

print(f'total len of raw data {len(data)}')

w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]

f = []
fl = []
fr = []


for img, lbl in data:
    if lbl == w:
        f.append([img, [0, 1, 0]])
    if lbl == wa:
        fl.append([img, [1, 0, 0]])
    if lbl == wd:
        fr.append([img, [0, 0, 1]])

# f = f[0:len(fr)]
# fl = fl[0:len(fr)]
shuffle(f)
shuffle(fl)
shuffle(fr)
f = f[0:len(fr)]
fl = fl[0:len(fr)]

print(len(f), len(fl), len(fr))

tot_data = f + fl + fr
# print('-------------', len(tot_data))
shuffle(tot_data)

# x_t = np.array([i[0] for i in tot_data]).reshape(-1, 420, 480, 1)
# y_t = np.array([i[1] for i in tot_data], dtype='float32')

train_data = tot_data[0:int(len(tot_data) * 0.8)]
test_data = tot_data[int(len(tot_data) * 0.8):]

x_train = np.array([i[0] for i in train_data])
x_test = np.array([i[0] for i in test_data])

y_train = np.array([i[1] for i in train_data])
y_test = np.array([i[1] for i in test_data])


# model = keras.Sequential([
#     keras.Input(shape=(180, 240, 3)),
#     layers.Conv2D(64, 3, activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),
#
#     layers.Conv2D(128, 3, activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),
#
#     layers.Conv2D(128, 3, activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(100, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.4),
#     layers.Dense(100, activation='relu'),
#     layers.Dense(3, activation='softmax'),
# ])


base_model = inception_v3.InceptionV3(include_top=False, weights=None, input_shape=(180, 240, 3))

x = base_model.output
x = keras.layers.GlobalAvgPool2D()(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
o = keras.layers.Dense(3, activation='softmax')(x)
model = keras.Model(base_model.inputs, o)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(0.001),
              metrics=['accuracy'],
              )
print(f'trainng on {len(x_train)}')

model.fit(x_train, y_train, epochs=35, validation_data=(x_test, y_test), batch_size=4)
model.save('roi-color-3var-Xception-0.001-35e')
