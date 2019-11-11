# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 17:31:53 2019

@author: 36236
"""

import math

class myNaiveBayes:
    
    def __init__(self):
        self.model = None
        
    @staticmethod  
    def mean(X):
        return sum(X) / float(len(X))
    def std(self, X):
        mea = self.mean(X)
        return math.sqrt(sum([pow(x-mea, 2) for x in X]) / float(len(X)))

    # 假设服从高斯分布，数据足够多，用各特征的均值和标准差来描述模型
    def fit(self, X, y):
        labels = list(set(y))
        data = {label:[] for label in labels}
        for f,  label in zip(X, y):
            data[label].append(f)
        self.model = {label: [(self.mean(i), self.std(i)) for i in zip(*value)] for label, value in data.items()}
    
    def gaussian_distribution(self, x, mean, std):
        exponent = math.exp(-(math.pow(x-mean, 2) / (x * math.pow(std, 2))))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent
    
    def probability(self, inputs):
        probablities = {}
        for label, value in self.model.items():
            probablities[label] = 1
            for i in range(len(value)):
                mean, std = value[i]
                probablities[label] *= self.gaussian_distribution(inputs[i], mean, std)
        return probablities
    
    def predict(self, X_test):
        return sorted(self.probability(X_test).items(), key=lambda x: x[-1])[-1][0]
    
    def score(self, X_test, y_test):
        tp = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                tp += 1
        return tp / float(len(X_test))
    