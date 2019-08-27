# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:19:06 2019

@author: 36236
"""
import numpy as np
from collections import Counter

class myKNN:
    def __init__(self, X_train, y_train, k=3, p=2):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k #value of k in KNN
        self.p = p #value of p in Minkowski distance
    
    def predict_point(self, X):
        #find k neighbors
        knn_list = []
        for i in range(self.k):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i])) #save distance and label
        for i in range(self.k, len(self.X_train)):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])
        #count
        label_list = [k[-1] for k in knn_list]
        count_pairs = Counter(label_list)
        predict_label = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]
        return predict_label
    
    def score(self, X_test, y_test):
        count = 0
        for X, y in zip(X_test, y_test):
            label = self.predict_point(X)
            if label == y:
                count += 1
        return count / len(X_test)
    
    def predict(self, X):
        y_predict = []
        for i in range(len(X)):
            y_predict.append(self.predict_point(X[i]))
        return np.array(y_predict)