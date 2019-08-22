# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 09:44:29 2019

@author: 36236
"""

import numpy as np

class myPreceptron(object):
    def __init__(self, data, label, lr=1):
        self.a = np.zeros([len(data), 1])
        self.b = 0
        self.lr = 1
        self.count = 0
        self.data = data
        self.label = label

    def model(self):
        gramMatrix = self.__get_gramMatrix(self.data)
        flag = True
        index = 0
        while flag:
            index += 1
            i = index % len(self.data) #All the points need to be judged
            self.__update(gramMatrix, self.label, i)
            if self.count == len(self.data):
                flag = False
        return np.sum(self.a * self.label.reshape(len(self.label),1) * self.data, axis=0), self.b       

    def __get_gramMatrix(self, data):
        return np.matmul(data, np.transpose(data))

    def __update(self, gramMatrix, label, i):
        sum = 0
        for j in range(len(self.a)):
            sum += self.a[j] * label[j] * gramMatrix[j][i] #sum of inner product 
        if label[i] * (sum + self.b) <= 0:
            self.a[i] += self.lr
            self.b += self.lr*label[i]
            self.count = 0 #As parameters are changed, all the points need to be rejudged
            return self.__update(gramMatrix, label, i)
        else:
            self.count += 1
        return
