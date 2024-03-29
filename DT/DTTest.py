# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:22:46 2019

@author: 36236
"""
import pandas as pd
from MyDT import myDT

data = [['青年', '否', '否', '一般', '否'],
           ['青年', '否', '否', '好', '否'],
           ['青年', '是', '否', '好', '是'],
           ['青年', '是', '是', '一般', '是'],
           ['青年', '否', '否', '一般', '否'],
           ['中年', '否', '否', '一般', '否'],
           ['中年', '否', '否', '好', '否'],
           ['中年', '是', '是', '好', '是'],
           ['中年', '否', '是', '非常好', '是'],
           ['中年', '否', '是', '非常好', '是'],
           ['老年', '否', '是', '非常好', '是'],
           ['老年', '否', '是', '好', '是'],
           ['老年', '是', '否', '好', '是'],
           ['老年', '是', '否', '非常好', '是'],
           ['老年', '否', '否', '一般', '否']]
columns = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
data_df = pd.DataFrame(data, columns=columns)
clf = myDT()
tree = clf.Fit(data_df) #可打印看一下，嵌套字典的形式
print(clf.Predict(['老年', '否', '否', '一般']))
