# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:57:25 2019

@author: 36236
"""

from MyANN import myANN
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
clf = myANN([784, 30 ,10])
clf.Train(training_data, 10, 10, 0.1, test_data)

