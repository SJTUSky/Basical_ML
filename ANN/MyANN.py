# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:57:13 2019

@author: 36236
"""

import numpy as np
import random

class myANN:
    
    def __init__(self, sizes):
        
        self.layerNumber = len(sizes) #网络层数，由输入得到
        self.sizes = sizes
        
        # 随机初始化权重 列表每个元素是array，即两层网络间的权重，大小=后一层网络神经元个数x前一层网络神经元个数
        self.W = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # 随机初始化偏移 列表每个元素是array，即两层网络间的偏移量，长度=后一层网络神经元个数x1
        self.b = [np.random.randn(y, 1) for y in sizes[1:]]
    
    # sigmoid 函数
    def Sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    
    
    # 进行训练，使用随机梯度下降，每次选部分样本来更新W和b
    def Train(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        
        n = len(training_data)
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
        
        for mini_batch in mini_batches:
            # 小批量样本来更新W和b
            self.Updata(mini_batch, learning_rate)
        
        # 测试样本准确率
        if test_data:
            n_test = len(test_data)
            print("Epoch {0}: {1} / {2}".format(j, self.Evaluate(test_data), n_test))
        else:
            print("Epoch {0} complete".format(j))
    
    # 用小批量样本更新W，b。使用BP得到的偏导数
    def Updata(self, mini_batch, learning_rate):
        # 空矩阵
        empty_W = [np.zeros(W.shape) for W in self.W]
        empty_b = [np.zeros(b.shape) for b in self.b]
        
        for x, y in mini_batch:
            delta_W, delta_b = self.BP(x, y)
            empty_W = [eW + dW for eW, dW in zip(empty_W, delta_W)]
            empty_b = [eb + db for eb, db in zip(empty_b, delta_b)]
        
        self.W = [W - (learning_rate / len(mini_batch)) * eW for W, eW in zip(self.W, empty_W)]
        self.b = [b - (learning_rate / len(mini_batch)) * eb for b, eb in zip(self.b, empty_b)]
    
    # sigmoid函数求导
    def SigmoidDe(self, z):
        return self.Sigmoid(z) * (1 - self.Sigmoid(z))
    
    # BP算法实现
    def BP(self, x, y):
         # 空矩阵
        delta_W = [np.zeros(W.shape) for W in self.W]
        delta_b = [np.zeros(b.shape) for b in self.b]
        # 神经元的输出值（激活后的值），的一层即样本输入
        activation = x
        activations = [x]
        # 储存神经元的输入值
        zs = []
        for W, b in zip(self.W, self.b):
            z = np.dot(W, activation) + b
            zs.append(z)
            print(z)
            print(len(z))
            activation = self.Sigmoid(z)
            activations.append(activation)
        # 损失对最后一层输入的导数
        delta = (y - activations[-1]) * self.SigmoidDe(zs[-1])
        #最后一层的参数
        delta_W[-1] = np.dot(delta, activations[-2].transpose())
        delta_b[-1] = delta
        # 循环计算前面的层的差值
        for i in range(2, self.layerNumber):
            z = zs[-i]
            sp = self.SigmoidDe(z)
            delta = np.dot(self.W[-i+1].transpose(), delta) * sp
            delta_W[-i] = np.dot(delta, activations[-i-1].transpose() )
            delta_b[-i] = delta
        return (delta_W, delta_b)
    
    # 由输入和参数计算输出
    def CalcalateOutput(self, a):
        
        # 把a当做上一层网络的输出y
        y = a
        
        for W, b in zip(self.W, self.b):
            y = self.Sigmoid(np.dot(W, y) + b)      
        
        return y
    
    # 评估结果
    def Evaluate(self, test_data):
        
        test_results = [(np.argmax(self.CalcalateOutput(x)), y) for (x, y) in test_data]
        num = sum(int(x == y) for (x, y) in test_results)
        
        return num