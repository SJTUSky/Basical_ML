# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 17:32:32 2019

@author: 36236
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from MyNaiveBayes import myNaiveBayes

#data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

#train and test data
data = np.array(df.iloc[:100, :])
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=47)

#model
clf = myNaiveBayes()
clf.fit(X_train, y_train)
print(clf.predict([4.4,  3.2,  1.3,  0.2]))
print(clf.score(X_test, y_test))
