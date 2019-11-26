# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:03:13 2019

@author: 36236
"""
import numpy as np

class mySVM:
    def __init__(self, max_inter=100, kernel='linear'):
        self.max_inter = max_inter
        self._kernel = kernel
    
    # 初始化SVM的一些参数值
    def InitArgs(self, X_train, y_train):
        self.m, self.n = X_train.shape # m,n分别为样本个数和特征个数
        self.X = X_train
        self.Y = y_train
        self.b = 0.0 #参数b
        self.alpha = np.ones(self.m) #拉格朗日乘子个数等于样本个数
        self.E = [self.CalculateE(i) for i in range(self.m)] #存放差值的列表
        self.C = 1.0 #松弛变量
    
    # 判断是否满足KKT条件，返回bool值
    def KKT(self, i):
        y_g = self.CalculateG(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else: 
            return y_g <= 1
    
    # g(x) 为样本x的预测值（未取符号的预测值）
    def CalculateG(self, i):
        g = self.b
        for j in range(self.m):
            g += self.alpha[j] * self.Y[j] * self.Kernel(self.X[i], self.X[j])
        return g
    
    # Ei为第i个样本预测值与真实值得差值
    def CalculateE(self, i):
        return self.CalculateG(i) - self.Y[i] 
    
    # 核函数
    def Kernel(self, x1, x2):
        if self._kernel == 'linear':
            # 线性核
            return sum(x1[k] * x2[k] for k in range(self.n))
        elif self._kernel == 'poly':
            # 多项式核
            return (sum(x1[k] * x2[k] for k in range(self.n)) + 1) ** 2
        return 0
    
    # 变量选择
    def InitAlpha(self):
        
        # 遍历所有满足0<alpha<C的点，检验是否满足KKT,若都满足则遍历整个训练集
        # 所有满足0<alpha<C的样本的序号
        satisfy_list = []
        for i in range(self.m):
            if 0 < self.alpha[i] < self.C:
                satisfy_list.append(i)
        # 不满足0<alpha<C的样本的序号
        non_satisfy_list = []
        for i in range(self.m):
            if i not in satisfy_list:
                non_satisfy_list.append(i)
        # 不满足0<alpha<C的样本放在后面，可以先遍历满足0<alpha<C的样本
        index_list = satisfy_list + non_satisfy_list
        
        for i in index_list:
            if self.KKT(i):
                continue
            #直到发现了一个不满足KKT的样本
            E1 = self.E[i]
            if E1 >= 0:
                j = self.E.index(min(self.E)) #E[x]的最小值的位置
            else:
                j = self.E.index(max(self.E)) #E[x]的最大值的位置
            return i, j
    
    # 训练，使用SMO算法
    def Fit(self, X_train, y_train):
        
        self.InitArgs(X_train, y_train)
        
        for t in range(self.max_inter):
            # 选择变量
            i1, i2 = self.InitAlpha()
            #计算边界
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.alpha[i2] - self.alpha[i1] + self.C)
            
            E1 = self.E[i1]
            E2 = self.E[i2]
            
            K11 = self.Kernel(self.X[i1], self.X[i1])
            K12 = self.Kernel(self.X[i1], self.X[i2])
            K22 = self.Kernel(self.X[i2], self.X[i2])
            
            # 计算η的值
            eta = K11 + K22 - 2 * K12
            if eta <= 0:
                break #eta开始为负后，不再进行优化，相当于一个停机条件
            
            #2个变量的y和alpha值
            y1 = self.Y[i1]
            y2 = self.Y[i2]
            alpha1_old = self.alpha[i1] 
            alpha2_old = self.alpha[i2] 
            
            #未经剪辑的新alpha值
            alpha2_new_unc = alpha2_old + y2 * (E1 - E2) / eta
            
            #alpha2值与边界比较，使其不超过边界
            if alpha2_new_unc > H:
                alpha2_new = H
            elif alpha2_new_unc < L:
                alpha2_new = L
            else:
                alpha2_new = alpha2_new_unc
            
            # alpha1的新值
            alpha1_new = alpha1_old +  y1 * y2 * (alpha2_old - alpha2_new)
            
            # b的新值
            b1_new = -E1 - y1 * K11 * (alpha1_new - alpha1_old) - y2 * K12 * (alpha2_new - alpha2_old) + self.b
            b2_new = -E2 - y1 * K12 * (alpha1_new - alpha1_old) - y2 * K22 * (alpha2_new - alpha2_old) + self.b
            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2
            
            #替换alpha值
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            # 更新b和E列表
            self.b = b_new
            self.E[i1] = self.CalculateE(i1) #b已经更新了，重新算E,书上是用支持向量，这里还是用所有向量
            self.E[i2] = self.CalculateE(i2)
    
    def Predict(self, data):
        g = self.b
        for j in range(self.m):
            g += self.alpha[j] * self.Y[j] * self.Kernel(data, self.X[j])
        if g > 0:
            return 1
        else:
            return -1 
    
    def Score(self, X_test, y_test):
        count = 0
        for i in range(len(X_test)):
            if self.Predict(X_test[i]) == y_test[i]:
                count += 1
        return count / len(X_test)