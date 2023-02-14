# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 21:27:18 2022

@author: 20050
"""



#Import the data

import pandas as pd
df=pd.read_csv("glass.csv")
df

df.dtypes
df.shape
df.info()
df.head()

# split the data as X and y variable
Y = df["Type"]
Y
X = df.iloc[:,0:9]
X

# Data Transformation--> standard scalar
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss_X = ss.fit_transform(X)

# Data Partition

from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test = train_test_split(ss_X,Y , test_size=0.3)

# fit the model

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2) # k =5 # p=2 --> Eucledian distance
knn.fit(X_train, Y_train)

# Prediction
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,y_pred_train)
ac1.round(2)*100

from sklearn.metrics import accuracy_score
ac2 = accuracy_score(Y_test,y_pred_test)
ac2.round(2)*100

train_accuracy = []
test_accuracy = []    

for i in range(5,20,2):
    knn = KNeighborsClassifier(n_neighbors=i, p=2)
    knn.fit(X_train, Y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)
    train_accuracy.append(accuracy_score(Y_train,y_pred_train).round(2))
    test_accuracy.append(accuracy_score(Y_test,y_pred_test).round(2))

d1 = pd.DataFrame(range(5,20,2))
d2 = pd.DataFrame(train_accuracy)
d3 = pd.DataFrame(test_accuracy)
    
pd.concat([d1,d2,d3],axis=1)

train_accuracy = []
test_accuracy = []    

for i in range(1,500,1):
    X_train, X_test,Y_train, Y_test = train_test_split(ss_X,Y , test_size=0.3, random_state=i)
    KNeighborsClassifier(n_neighbors=13, p=2)
    knn.fit(X_train, Y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)
    train_accuracy.append(accuracy_score(Y_train,y_pred_train).round(2))
    test_accuracy.append(accuracy_score(Y_test,y_pred_test).round(2))

import numpy as np    
np.mean(train_accuracy).round(3)
np.mean(test_accuracy).round(3)
    










































































