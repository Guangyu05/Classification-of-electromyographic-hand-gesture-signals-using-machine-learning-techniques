# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:18:05 2018

@author: Guangyu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 20:41:26 2018

@author: Guangyu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 23:14:06 2018

@author: Guangyu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 00:56:59 2018

@author: Guangyu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:07:39 2018

@author: Guangyu
"""
#This model is used to increase the robustness of the model using 10% of the cropped data as the input data
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
import time

data1 = sio.loadmat('XTrain_11_12_2018.mat')
data1 = data1.get('XTrain')
data1 = np.float32(data1)
data1 = data1.transpose((3,0,1,2))

with h5py.File('XTest_11_12_2018.mat') as f:
    data2 = f['XTest'][:]
    data2 = np.float32(data2)
data2 = data2.transpose((0,3,2,1))

ytrain = sio.loadmat('YTrain_11_12_2018.mat')
data_train_label = ytrain.get('YTrain')
ytrain1 = keras.utils.to_categorical(data_train_label,11)
ytrain1 = np.delete(ytrain1,0,axis=1)

ytest = sio.loadmat('YTest_11_12_2018.mat')
data_test_label = ytest.get('YTest')
ytest1 = keras.utils.to_categorical(data_test_label,11)
ytest1 = np.delete(ytest1,0,axis=1)

ytest_target = sio.loadmat('data_target_160.mat')
ytest_target= ytest_target.get('data_target')
ytest_target1 = []
for i in ytest_target:
    temp = i-1
    ytest_target1.append(temp)

start = time.clock()
input_img = Input(shape=(500,2,1))
x = Conv2D(30, kernel_size=(2, 2),activation='relu',padding = 'same')(input_img)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D((2,1), padding = 'same')(x)
x = Conv2D(20,(2,1),activation='relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)
encoded = MaxPooling2D((2,1), padding = 'same')(x)
x = ReLU()(encoded)
x = BatchNormalization()(x)
x = Conv2D(20,(2,1), padding = 'same')(x)
x = UpSampling2D((2,1))(x)
x = ReLU()(x)
x = BatchNormalization()(x)
x = Conv2D(20,(2,2),activation='relu', padding = 'same')(x)
x = UpSampling2D((2,1))(x)
decoded = Conv2D(1,(2,2),activation='relu', padding = 'same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='mean_squared_error',
                    metrics = ['accuracy'])
autoencoder.fit(data1, data1,
          batch_size=128,
          epochs=8,
          verbose=1,
          shuffle=True)
#score = autoencoder.evaluate(data2, data2, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

encoder = Model(input_img, encoded)
encoder.summary()

layer_index = 9
intermediate_layer_model = Model(inputs=autoencoder.input,
                                 outputs=autoencoder.get_layer(index = layer_index).output)
intermediate_output = intermediate_layer_model.predict(data1)
intermediate_output_2 = intermediate_layer_model.predict(data2)

a = intermediate_output.shape
a = a[1:]


model = Sequential()
model.add(Conv2D(30, kernel_size=(2, 2),activation='relu', input_shape=a))
model.add(BatchNormalization())
model.add(ReLU())
#model.add(Dropout(0.10))
model.add(MaxPooling2D(pool_size=(2,1),strides=(2,1)))
model.add(Conv2D(50, (5,1),activation='relu'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2,1),strides=(2,1)))
#model.add(Dropout(0.10))
model.add(Conv2D(100, (5,1),activation='relu'))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2,1),strides=2))
model.add(Conv2D(30, (8,1), activation='relu'))
model.add(BatchNormalization())
model.add(ReLU())
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
          batch_size=256,
          epochs=15, 
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
elapsed = (time.clock() - start)
with h5py.File('XTest_500.mat') as f:
    data2 = f['XTest'][:]
    data2 = np.float32(data2)
data2 = data2.transpose((0,3,2,1))


intermediate_output_2 = intermediate_layer_model.predict(data2)
y_test = model.predict(intermediate_output_2)
score = model.evaluate(intermediate_output_2, ytest1, verbose=0)

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

#model.save('AE_CNN_best_best.h5')
#intermediate_layer_model.save('intermediate_layer_model_best_best.h5')
#autoencoder.save('AE_best_best.h5')

##################################rf1
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import timeit
import numpy as np
import scipy.io as sio  
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
import pydotplus 
from numpy import ones
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib 
data1 = data1.reshape((62432,1000))
data2 = data2.reshape((312160,1000))
data_train_label = np.ravel(data_train_label)
data_test_label = np.ravel(data_test_label)
ytest_target = sio.loadmat('data_target_160.mat')
ytest_target= ytest_target.get('data_target')
ytest_target1 = []
for i in ytest_target:
    temp = i
    ytest_target1.append(temp)
    
############################################rf
clf_rf = RandomForestClassifier()
clf_rf.fit(data1, data_train_label)
y_pred_rf = clf_rf.predict(data2)
acc_rf = accuracy_score(data_test_label, y_pred_rf)
print(acc_rf)
y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_pred_rf[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy_rf = a/len(ytest_target1)  
print(final_accuracy_rf)  
joblib.dump(clf_rf, 'save/clf_rf.pkl')
clf_rf = joblib.load('save/clf_rf.pkl')
#0.9125
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

y_pred_svm = np.zeros((312160,1))
for i in range(225819,len(data2)):
    a = data2[i].reshape(-1,1)
    y_pred_svm[i] = clf_svm.predict(a.transpose())

y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_pred_svm[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])
    
y_test2 = np.ravel(y_test2)
y_test2 = np.uint8(y_test2)

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy_svm = a/len(ytest_target1)  
print(final_accuracy_svm)  
clf_svm = joblib.load('clf_svm.pkl') 
#72.5%
##################################dt
clf_dt = tree.DecisionTreeClassifier()
clf_dt.fit(data1, data_train_label)
y_pred_decisiontree = clf_dt.predict(data2)
score=clf_dt.score(data2,data_test_label)
print("%f"%score)

y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_pred_decisiontree[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy_decision_tree = a/len(ytest_target1)  
print(final_accuracy_decision_tree)
joblib.dump(clf, 'save/clf_dt.pkl')
clf_dt = joblib.load('save/clf_dt.pkl')
y_pred_decisiontree = clf_dt.predict(data2)
score=clf_dt.score(data2,data_test_label)
print("%f"%score)
#0.90
##################################knn
clf_knn = KNeighborsClassifier()
clf_knn.fit(data1, data_train_label)
y_pred_knn = np.zeros((312160,1))
#y_pred_knn = np.load('y_pred_knn.npy')

for i in range(0,len(data2)):
    a = data2[i].reshape(-1,1)
    y_pred_knn[i] = clf_knn.predict(a.transpose())
    
acc_knn = accuracy_score(data_test_label, y_pred_knn)
print(acc_knn)
y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_pred_knn[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy_knn = a/len(ytest_target1)  
print(final_accuracy_knn) 

#83.125
##################################################nn
y_train = np.ravel(data_train_label)
y_test = np.ravel(data_test_label)
clf_nn = MLPClassifier(hidden_layer_sizes=(21,21,21,),verbose=1,activation='logistic')
clf_nn.fit(data1, y_train)
y_pred_nn = clf_nn.predict(data2)
acc_nn = clf_nn.score(data2,y_test)
print(acc_nn)

y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_pred_nn[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy_nn = a/len(ytest_target1)  
print(final_accuracy_nn) 
#save Model
joblib.dump(clf_nn, 'save/clf_nn.pkl')

#read Model
clf_nn = joblib.load('save/clf.pkl')
#0.1
####################################################sgd
clf_sgd = SGDClassifier()
clf_sgd.fit(data1, data_train_label)
y_pred_sgd = clf_sgd.predict(data2)
acc_sgd = accuracy_score(data_test_label, y_pred_sgd)
print("stochastic gradient descent accuracy: ",acc_sgd)

##########################################################regression
y_train = np.ravel(data_train_label)
y_test = np.ravel(data_test_label)
clf = LogisticRegression()
clf.fit(data1,y_train)
lr_test_sc=clf.score(data2,y_test)
print("regression: ",lr_test_sc)
y_pred_reg = clf.predict(data2)

y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_pred_reg[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy_reg = a/len(ytest_target1)  
print(final_accuracy_reg) 
##########################################nb
clf_gnb = GaussianNB()
clf_gnb.fit(data1,y_train)
y_pred_gnb = clf_gnb.predict(data2)
acc_gnb = clf_gnb.score(data2, y_test)
print("nb accuracy: ",acc_gnb)

y_test2 = []
for i in range(1,161):     
    a = scipy.stats.mode(y_pred_gnb[(i-1)*1951:i*1951],axis=0)
    y_test2.append(a[0])

a = 0    
for n in range(len(ytest_target1)):
    if ytest_target1[n] == y_test2[n]:
        a = a+1
final_accuracy_gnb = a/len(ytest_target1)  
print(final_accuracy_gnb) 

from sklearn.metrics import multilabel_confusion_matrix
y_true = ytest_target1
y_pred = y_test2
multilabel_confusion_matrix(y_true, y_pred)