# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:15:14 2019

@author: 36236
"""
import numpy as np
from math import sqrt

class Node:
    def __init__(self, data, depth=0, lchild=None, rchild=None):
        self.data = data
        self.depth = depth
        self.lchild = lchild #left node
        self.rchild = rchild #right node
        
class myKdTree:
    def __init__(self):
        self.KdTree = None
        self.n = 0
        self.nearest =None
        
    def create(self, dataSet, depth=0):
        if len(dataSet) > 0:
            m, n  = np.shape(dataSet)
            self.n = n - 1 #As the last column is the label
            axis = depth % self.n 
            mid = int(m / 2)
            dataSetCopy = sorted(dataSet, key=lambda x: x[axis])
            node = Node(dataSetCopy[mid], depth)
            if depth == 0:
                self.KdTree = node
            node.lchild = self.create(dataSetCopy[:mid], depth+1)
            node.rchild = self.create(dataSetCopy[mid+1:], depth+1)
            return node
        return None
    
    def preOrder(self, node):
        if node is not None:
            print(node.depth, node.data)
            self.preOrder(node.lchild)
            self.preOrder(node.rchild)
            
    def search(self, x, k=1):
        nearest = []
        for i in range(k):
            nearest.append([-1, None])
        self.nearest = np.array(nearest)
        
        def recurve(node):
            if node is not None:
                axis = node.depth % self.n
                daxis = x[axis] - node.data[axis]
                #move to leaf node
                if daxis < 0:
                    recurve(node.lchild)
                else:
                    recurve(node.rchild)
                    
                dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(x, node.data)))
                for i, d in enumerate(self.nearest):
                    if d[0] < 0 or dist < d[0]:
                        self.nearest = np.insert(self.nearest, i, [dist, node], axis=0)
                        self.nearest = self.nearest[:-1]
                        break
                    
                n = list(self.nearest[:, 0]).count(-1)
                if self.nearest[-n-1, 0] > abs(daxis): 
                    #This mean that the point of intersection is existing
                    #Search another child node of the father node
                    if daxis < 0:
                        recurve(node.rchild)
                    else:
                        recurve(node.lchild)
        
        recurve(self.KdTree)
        node_list = self.nearest[:,1]
        label_list = []
        for k in node_list:
            label_list.append(k.data[-1])
        predict_label = max(set(label_list), key=label_list.count)
        
        return self.nearest, predict_label
        
        
        
        