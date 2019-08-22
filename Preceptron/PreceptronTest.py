# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:19:33 2019

@author: 36236
"""

import numpy as np
from MyPreceptron import myPreceptron
import matplotlib.pyplot as plt

def main():
    
     #data
    data = np.array([[3,3], [3,1], [1,3], [1,1], [2,1], [1,2]])
    label = np.array([1,1,1,-1,-1,-1])
    
    #model
    w, b = myPreceptron(data, label).model()
    print('W %s, \n %s.\n' % (w, b))
    
    #draw point
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(len(data)):
        if label[i] == 1:
            ax.scatter(data[i,0], data[i,1], c='r')
        else:
            ax.scatter(data[i,0], data[i,1], c='g')
    plt.show()
    #draw line
    x = np.arange(0,4.5,0.5)
    y = (-b/w[1]-w[0]/w[1]*x)
    ax.plot(x,y,c='b')

if __name__ == '__main__':
    main()