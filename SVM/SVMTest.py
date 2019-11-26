# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:03:33 2019

@author: 36236
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from MySVM import mySVM

#data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

#train and test data
data = np.array(df.iloc[:100, [0,1,-1]])
for i in range(len(data)):
    if data[i,-1] == 0:
        data[i,-1] = -1
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# 模型
clf = mySVM(max_inter=200, kernel='linear')
clf.Fit(X_train, y_train)
result = clf.Score(X_test, y_test)
print(result)
