# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 19:49:16 2019

@author: Guangyu
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 02:43:04 2019

@author: Guangyu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 00:06:32 2019

@author: Guangyu
"""



import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, ReLU, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import h5py
import numpy as np
import struct
import scipy
from scipy import stats
import scipy.io as sio

import numpy as np
#from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#from nolearn.dbn import DBN
import timeit
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio  
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC  
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
import pydotplus 
from numpy import ones
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.naive_bayes import GaussianNB


data1 = sio.loadmat('XTrain_original_data.mat')
data1 = data1.get('XTrain')
data1 = data1.reshape((320,40000))

data2 = sio.loadmat('XTest_original_data.mat')
data2 = data2.get('XTest')
data2 = data2.reshape((160,40000))

ytrain = sio.loadmat('YTrain_same_with_NN_4_160.mat')
data_train_label = ytrain.get('YTrain')

ytest = sio.loadmat('YTest_same_with_NN_4_160.mat')
data_test_label = ytest.get('YTest')


ytest_target = sio.loadmat('data_target_160.mat')
ytest_target= ytest_target.get('data_target')
ytest_target1 = []
for i in ytest_target:
    temp = i
    ytest_target1.append(temp)
  
#####################################old nn
model = Sequential()
model.add(Dense(units=200,activation='relu', input_dim=4))
model.add(Dense(units=200,activation='relu'))
model.add(Dense(units=200,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.compile(optimizer='sgd', loss='mean_squared_error',
                    metrics = ['accuracy'])
model.fit(data1, ytrain1,
          batch_size=128,
          epochs=100, 
          verbose=1,
          shuffle=True)

from neupy import algorithms
lmnet = algorithms.LevenbergMarquardt((4, 10, 10),show_epoch=1)
lmnet.train(data1, ytrain1)



##################################rf1
data_train_label = np.ravel(data_train_label)
clf_rf = RandomForestClassifier()
clf_rf.fit(data1, data_train_label)
y_pred_rf = clf_rf.predict(data2)
acc_rf = accuracy_score(data_test_label, y_pred_rf)
print(acc_rf)
acc_rf_train = clf_rf.score(data1, data_train_label)
print(acc_rf_train)

################################svm
y_train = np.ravel(data_train_label)
y_test = np.ravel(data_test_label)
clf_svm = SVC(kernel='rbf')  
clf_svm.fit(data1, y_train)
y_pred_svm = clf_svm.predict(data2)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(acc_svm)
acc_svm_train = clf_svm.score(data1, y_train)
print(acc_svm_train)
acc_svm_test = clf_svm.score(data2, y_test)
print(acc_svm_test)

##################################dt
clf = tree.DecisionTreeClassifier()
clf.fit(data1, data_train_label)
y_pred_decisiontree = clf.predict(data2)
score=clf.score(data2,data_test_label)
print("%f"%score)
score_train=clf.score(data1, data_train_label)
print("%f"%score_train)


dot_data = tree.export_graphviz(clf, out_file=None, 
                     filled=True, rounded=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())
graph.write_jpg("dt1.jpg")


##################################knn
clf_knn = KNeighborsClassifier()
clf_knn.fit(data1, data_train_label)
y_pred_knn = clf_knn.predict(data2)
acc_knn = accuracy_score(data_test_label, y_pred_knn)
print(acc_knn)
acc_knn_train = clf_knn.score(data1, data_train_label)
print(acc_knn_train)

##################################################nn
y_train = np.ravel(data_train_label)
y_test = np.ravel(data_test_label)
clf_nn = MLPClassifier(hidden_layer_sizes=(21,21,21,),verbose=0,activation='relu')
clf_nn.fit(data1, y_train)
#y_pred_nn = clf_nn.predict(data2)
acc_nn = clf_nn.score(data2,y_test)
print(acc_nn)
acc_nn_train = clf_nn.score(data1, y_train)
print(acc_nn_train)
#0.1
####################################################sgd
clf_sgd = SGDClassifier()
clf_sgd.fit(data1, data_train_label)
y_pred_sgd = clf_sgd.predict(data2)
acc_sgd = clf_sgd.score(data2, data_test_label)
print("stochastic gradient descent accuracy: ",acc_sgd)

##########################################################regression
y_train = np.ravel(data_train_label)
y_test = np.ravel(data_test_label)
clf = LogisticRegression()
clf.fit(data1,y_train)
lr_test_sc=clf.score(data2,y_test)
print("regression: ",lr_test_sc)
lr_train_sc=clf.score(data1,y_train)
print("regression: ",lr_train_sc)
##########################################nb
clf_gnb = GaussianNB()
clf_gnb.fit(data1,y_train)
y_pred_gnb = clf_gnb.predict(data2)
acc_gnb = clf_gnb.score(data2, y_test)
print("nb accuracy: ",acc_gnb)
acc_gnb_train = clf_gnb.score(data1, y_train)
print("nb accuracy: ",acc_gnb_train)
##############################################AE+CNN
data1 = sio.loadmat('XTrain_original_data.mat')
data1 = data1.get('XTrain')
data1 = data1.reshape((320,20000,2,1))

data2 = sio.loadmat('XTest_original_data.mat')
data2 = data2.get('XTest')
data2 = data2.reshape((160,20000,2,1))

input_img = Input(shape=(20000,2,1))
x = Conv2D(30, kernel_size=(50, 2),activation='relu',padding = 'same')(input_img)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D((4,1), padding = 'same')(x)
x = Conv2D(20,(20,1),activation='relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
encoded = MaxPooling2D((4,1), padding = 'same')(x)
x = ReLU()(encoded)
x = BatchNormalization()(x)
x = Conv2D(20,(20,1), padding = 'same')(x)
x = UpSampling2D((4,1))(x)
x = ReLU()(x)
x = BatchNormalization()(x)
x = Conv2D(20,(50,2),activation='relu', padding = 'same')(x)
x = UpSampling2D((4,1))(x)
decoded = Conv2D(1,(2,2),activation='relu', padding = 'same')(x)


autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='mean_squared_error',
                    metrics = ['accuracy'])
#earlystop=keras.callbacks.EarlyStopping(monitor='acc', min_delta=0, patience=3, verbose=0, mode='min')
autoencoder.fit(data1, data1,
          batch_size=15,
          epochs=25,
          verbose=1,
          shuffle=True)
          #callbacks=[earlystop])


encoder = Model(input_img, encoded)
encoder.summary()

layer_index = 9
intermediate_layer_model = Model(inputs=autoencoder.input,
                                 outputs=autoencoder.get_layer(index = layer_index).output)
intermediate_output = intermediate_layer_model.predict(data1)
intermediate_output_2 = intermediate_layer_model.predict(data2)

a = intermediate_output.shape
a = a[1:]


ytrain1 = keras.utils.to_categorical(data_train_label,11)
ytrain1 = np.delete(ytrain1,0,axis=1)
ytest1 = keras.utils.to_categorical(data_test_label,11)
ytest1 = np.delete(ytest1,0,axis=1)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, ReLU
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

model = Sequential()
model.add(Conv2D(20, kernel_size=(10, 2),activation='relu',input_shape=a))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.10))
model.add(MaxPooling2D(pool_size=(2,1),strides=(2,1)))
model.add(Flatten())
model.add(Dropout(0.10))
#model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='mse', 
               optimizer = 'adam',
               metrics = ['accuracy'])
#optimizer = keras.optimizers.Adam(lr=0.001,beta_1=0.90,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False), 
#optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
model.summary()

model.fit(intermediate_output, ytrain1,
          batch_size=40,
          epochs=50, 
          verbose=1,
          shuffle=True,
          validation_data=(intermediate_output_2, ytest1))
          #validation_data=(intermediate_output_2, ytest1))

model.compile(loss='mse', 
               optimizer = 'sgd',
               metrics = ['accuracy'])
#optimizer = keras.optimizers.Adam(lr=0.001,beta_1=0.90,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False), 
#optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
#model.summary()

model.fit(intermediate_output, ytrain1,
          batch_size=40,
          epochs=20, 
          verbose=1,
          shuffle=True,
          validation_data=(intermediate_output_2, ytest1))

keras.callbacks.EarlyStopping(monitor = 'val_acc',
                              min_delta=0.1,
                              patience=3,
                              verbose=0,mode='auto')

intermediate_output_2 = intermediate_layer_model.predict(data2)
#y_test = model.predict(intermediate_output_2)
score = model.evaluate(intermediate_output_2, ytest1, verbose=0)
print(score)

intermediate_output_train = intermediate_layer_model.predict(data1)
#y_train_CAE = model.predict(intermediate_output_train)
score = model.evaluate(intermediate_output_train, ytrain1, verbose=0)
print(score)

####################################################CNN
data1 = sio.loadmat('XTrain_original_data.mat')
data1 = data1.get('XTrain')
data1 = data1.reshape((320,20000,2,1))

data2 = sio.loadmat('XTest_original_data.mat')
data2 = data2.get('XTest')
data2 = data2.reshape((160,20000,2,1))

ytrain1 = keras.utils.to_categorical(data_train_label,11)
ytrain1 = np.delete(ytrain1,0,axis=1)
ytest1 = keras.utils.to_categorical(data_test_label,11)
ytest1 = np.delete(ytest1,0,axis=1)

input_img = Input(shape=(20000,2,1))
x = Conv2D(30, kernel_size=(20, 2),activation='relu')(input_img)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D((4,1))(x)
x = Conv2D(20,(20,1),activation='relu')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D((4,1))(x)
x = ReLU()(x)
x = BatchNormalization()(x)
x = Conv2D(20,(20,1))(x)
x = MaxPooling2D((4,1))(x)
x = ReLU()(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
x_final = Dense(10,activation='softmax')(x)

cnn = Model(input_img, x_final)
cnn.summary()
cnn.compile(optimizer='adam', loss='mean_squared_error',
                    metrics = ['accuracy'])
cnn.fit(data1, ytrain1,
          batch_size=20, #10
          epochs=50,    #25
          verbose=1,
          shuffle=True,
          validation_data=(data2, ytest1))
cnn.compile(optimizer='sgd', loss='mean_squared_error',
                    metrics = ['accuracy'])
cnn.fit(data1, ytrain1,
          batch_size=10,
          epochs=5,
          verbose=1,
          shuffle=True,
          validation_data=(data2, ytest1))
#earlystop=keras.callbacks.EarlyStopping(monitor='acc', min_delta=0, patience=3, verbose=0, mode='min')

score_train_cnn = cnn.evaluate(data1,ytrain1)
print(score_train_cnn)
score_test_cnn = cnn.evaluate(data2,ytest1)
print(score_test_cnn)
