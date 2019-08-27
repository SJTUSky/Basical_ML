# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:19:27 2019

@author: 36236
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from MyKNN import myKNN

#data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

#distribution of the data
plt.figure(figsize=(9, 8))
plt.subplot(221)
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], c='b')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], c='g')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('original data')

#train and test data
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

#classification
clf = myKNN(X_train, y_train)
print("Test_score: {}".format(clf.score(X_test, y_test)))
test_point = [6.0, 3.0]
print("label of test point: {}".format(clf.predict_point(test_point)))

#figures
plt.subplot(222)
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], c='b', label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], c='g', label='1')
plt.plot(test_point[0], test_point[1], 'bo', c='r', label='test_point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.title('result of test point by KNN(n=3, p=2)')

plt.subplot(223)
for i in range(len(X_train)):
    if y_train[i] == 0:
        plt.scatter(X_train[i,0], X_train[i,1], c='b')
    else:
        plt.scatter(X_train[i,0], X_train[i,1], c='g')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('training data')

plt.subplot(224)
y_predict = clf.predict(X_test)
for i in range(len(X_test)):
    if y_test[i] == 0 and y_predict[i] == 0:
        plt.scatter(X_test[i,0], X_test[i,1], marker = "o", c='b')
    elif y_test[i] == 0 and y_predict[i] == 1:
        plt.scatter(X_test[i,0], X_test[i,1], marker = "x", c='b')
    elif y_test[i] == 1 and y_predict[i] == 0:
        plt.scatter(X_test[i,0], X_test[i,1], marker = "x", c='g')
    else:
        plt.scatter(X_test[i,0], X_test[i,1], marker = "o", c='g')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('test data')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.25)