# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:15:40 2019

@author: 36236
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from MyKdTree import myKdTree

#data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

#train and test data
data = np.array(df.iloc[:100, [0, 1, -1]])
train, test = train_test_split(data, test_size=0.4, random_state=47)
x_0 = np.array([x0 for i, x0 in enumerate(train) if train[i][-1] == 0])
x_1 = np.array([x1 for i, x1 in enumerate(train) if train[i][-1] == 1])

#model
clf = myKdTree()
clf.create(train)
clf.preOrder(clf.KdTree)

acc = 0
for x in test[:4]:
    # distribution of the data
    plt.figure()
    plt.scatter(x_0[:, 0], x_0[:, 1], c='pink', label='0')
    plt.scatter(x_1[:, 0], x_1[:, 1], c='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.scatter(x[0], x[1], c='r', marker='x', label='test point')
    # predict
    k_nearbor, predict_label = clf.search(x[:-1], 5)
    if predict_label == x[-1]:
        acc += 1
    print("test:")
    print(x, "predict:", predict_label)
    print("nearest:")
    for k in k_nearbor:
        print(k[1].data, "dist:", k[0])
        plt.scatter(k[1].data[0], k[1].data[1], c='black', marker='+')
    plt.legend()
    
acc /= len(test[:4])
print("accuracy:", acc)    