# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 12:30:22 2018

@author: Guangyu
"""

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, ReLU, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
import h5py
import numpy as np
import struct
import scipy
from scipy import stats
import scipy.io as sio
import pandas as pd
#Data processing

import scipy.io as sio
import pandas as pd
data1 = sio.loadmat('XTrain_same_with_NN.mat')
data1 = data1.get('XTrain')
data1 = np.float32(data1)
data1 = data1.transpose((3,0,1,2))

data2 = sio.loadmat('XTest_same_with_NN.mat')
data2 = data2.get('XTest')
data2 = np.float32(data2)
data2 = data2.transpose((3,0,1,2))

ytrain = sio.loadmat('YTrain_same_with_NN.mat')
data_train_label = ytrain.get('YTrain')
ytrain1 = keras.utils.to_categorical(data_train_label,11)
ytrain1 = np.delete(ytrain1,0,axis=1)

ytest = sio.loadmat('YTest_same_with_NN.mat')
data_test_label = ytest.get('YTest')
ytest1 = keras.utils.to_categorical(data_test_label,11)
ytest1 = np.delete(ytest1,0,axis=1)

ytest_target = sio.loadmat('data_target_160.mat')
ytest_target= ytest_target.get('data_target')
ytest_target1 = []
for i in ytest_target:
    temp = i-1
    ytest_target1.append(temp)

input_img = Input(shape=(2,2,1))
x = Conv2D(200, kernel_size=(2, 2),activation='relu',padding = 'same')(input_img)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv2D(20,(2,2),activation='relu', padding = 'same')(x)
x = BatchNormalization()(x)
encoded = ReLU()(x)
x = Conv2D(20,(2,2), padding = 'same')(encoded)
x = ReLU()(x)
x = BatchNormalization()(x)
x = Conv2D(5,(2,2),activation='relu', padding = 'same')(x)
decoded = Conv2D(1,(2,2),activation='relu', padding = 'same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='Adam', loss='mean_squared_error',
                    metrics = ['accuracy'])
autoencoder.fit(data1, data1,
          batch_size=512,
          epochs=30,
          verbose=1,
          shuffle=True)
#score = autoencoder.evaluate(data2, data2, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

encoder = Model(input_img, encoded)
encoder.summary()

layer_index = 7
intermediate_layer_model = Model(inputs=autoencoder.input,
                                 outputs=autoencoder.get_layer(index = layer_index).output)
intermediate_output = intermediate_layer_model.predict(data1)
intermediate_output_2 = intermediate_layer_model.predict(data2)

a = intermediate_output.shape
a = a[1:]

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, ReLU
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

model = Sequential()
model.add(Conv2D(30, kernel_size=(2, 2),activation='relu', input_shape=a,padding = 'same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Conv2D(50, (2,2),activation='relu',padding = 'same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Conv2D(50, (2,2),activation='relu',padding = 'same'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Flatten())
#model.add(Dropout(0.10))
#model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='mse', 
               optimizer = 'Adam',
               metrics = ['accuracy'])
#optimizer = keras.optimizers.Adam(lr=0.001,beta_1=0.90,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False), 
#optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
model.summary()

model.fit(intermediate_output, ytrain1,
          batch_size=256,
          epochs=30, 
          verbose=1,
          shuffle=True)
          #validation_data=(intermediate_output_2, ytest1))

model.compile(loss='mse', 
               optimizer = 'sgd',
               metrics = ['accuracy'])
#optimizer = keras.optimizers.Adam(lr=0.001,beta_1=0.90,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False), 
#optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
#model.summary()

model.fit(intermediate_output, ytrain1,
          batch_size=256,
          epochs=5, 
          verbose=1,
          shuffle=True,
          validation_data=(intermediate_output_2, ytest1))

keras.callbacks.EarlyStopping(monitor = 'val_acc',
                              min_delta=0.1,
                              patience=3,
                              verbose=0,mode='auto')

#y = model.predict(intermediate_output_2)

with h5py.File('XTest_noise_2e6.mat') as f:
    data2 = f['XTest'][:]
    data2 = np.float32(data2)
data2 = data2.transpose((0,3,2,1))


intermediate_output_2 = intermediate_layer_model.predict(data2)
y_test = model.predict(intermediate_output_2)
score = model.evaluate(intermediate_output_2, ytest1, verbose=0)
print(score)
intermediate_output_2_train = intermediate_layer_model.predict(data1)
y_train = model.predict(intermediate_output_2_train)
score_train = model.evaluate(intermediate_output_2_train, y_train, verbose=0)
print(score_train)

y_test1 = []

for r in y_test:
    r = r.tolist()
    index = r.index(max(r))
    y_test1.append(index)
#    if i > y_test.shape[0]:
#        break
   
y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_test1[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy = a/len(ytest_target1)  
print(final_accuracy)      




