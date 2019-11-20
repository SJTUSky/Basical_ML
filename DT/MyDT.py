# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:21:53 2019

@author: 36236
ID3决策树
建好的树是一个类，有各种属性，leaf描述是否为末端节点，label在是末端节点时即其标签，特征名、特征位置
树也可以看做一个字典，像输出的那样，一个个键值对
特殊的是tree，其值是空（叶节点）或者多个子树，又是一个新的字典
   字典的键为特征的不同取值，例如，键为‘是’，值为一棵子树，另一个键为'否'，值为另一棵子树
   子树再分别是父树一样的字典，包含不同属性，可看做一个个键值对
例子
一个作为末端节点的子树
{'label': '否'
 'featureName': None
 'tree': {}
}
其父节点包含的树
{'label': None
 'featureName': '有工作'
 'tree': {'否'：{'label': '否'
                 'featureName': None
                 'tree': {}
                }
          '是': {'label': '是'
                 'featureName': None
                 'tree': {}
                }
         }
}
注意前面的'否'是指特征的值，可以理解为是否满足条件。后面的'否'是标签

"""
from math import log2
import pandas as pd
import numpy as np

class Node:
    def __init__(self, leaf=True, label = None, feature_name  = None, feature = None):
        self.leaf = leaf
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {'label': self.label, 'featureName': self.feature_name, 'tree': self.tree}
    def __repr__(self):
        return '{}'.format(self.result)
    def Add_node(self, val, node):
        # tree是一个字典，键val即特征的值，例如，键为‘是’，值为一棵树，另一个键为'否'，值为另一棵树
        self.tree[val] = node
    def Predict(self, features):
        # 往下面的子树一点点找，直到找到叶节点，其label就是预测的label
        if self.leaf is True:
            return self.label
        return self.tree[features[self.feature]].Predict(features)

class myDT:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}
    @staticmethod
    # 计算经验熵
    def Entropy(data):
        labelSet = {}
        for i in range(len(data)):
            label = data[i][-1]
            if label not in labelSet.keys():
                labelSet[label] = 0
            labelSet[label] += 1
        entropy = -sum([(p/len(data))*log2(p/len(data)) for p in labelSet.values()])
        return entropy
    # 计算经验条件熵
    def Cond_Entropy(self, data, axis = 0):
        featureSet = {}
        for i in range(len(data)):
            feature = data[i][axis]
            if feature not in featureSet.keys():
                featureSet[feature] = []
            featureSet[feature].append(data[i])
        cond_entropy = sum([(len(p)/len(data)) * self.Entropy(p) for p in featureSet.values()])
        return cond_entropy
    # 计算data的信息增益
    def Info_gain_train(self, data):
        hD = self.Entropy(data)
        features = []
        for c in range(len(data[0]) - 1):
            c_info_gain = hD - self.Cond_Entropy(data, axis=c)
            features.append((c, c_info_gain))
        best_feature = max(features, key = lambda x: x[-1])
        return best_feature
    # 训练数据
    def Train(self, train_data):
        y_train = train_data.iloc[:, -1] # 所有的标签
        features = train_data.columns[:-1] # 仅包含特征的值的数据
        # D为数据集，A为特征集
        # 1 若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        if len(y_train.value_counts()) == 1:
            # 判断y_train中有几个不同的值
            return Node(leaf=True, label=y_train.iloc[0])
        # 2 若A为空，则T为单节点树，将D中实例数最多的类Ck作为该节点的类标记，返回T
        if len(features) == 0:
            return Node(leaf=True, label=y_train.value_counts().sort_values(ascending=False).index[0])
        # 3 否则，计算A中各特征对D的信息增益，选择信息增益最大的特征Ag
        max_feature, max_info_gain = self.Info_gain_train(np.array(train_data)) #feature值为第几个特征
        max_feature_name = features[max_feature]
        # 4 若Ag的信息增益小于阈值epsilon，则T为单节点树，并将D中实例数最多的类Ck作为该节点的类标记，返回T
        if max_info_gain < self.epsilon:
            return Node(leaf=True, label=y_train.value_counts().sort_values(ascending=False).index[0])
        # 5 否则，对Ag的每一可能ai，依Ag=ai将D分为若干非空子集Di，将Di中实例数最大的类作为标记，构建子节点
        node_tree = Node(leaf=False, feature_name=max_feature_name, feature=max_feature)
        feature_list = train_data[max_feature_name].value_counts().index # 该特征的所有取值
        for f in feature_list:
            # 获得当前特征取值为f的样本子集Di，并去掉当前特征Ag
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)
            # 6 以Di为训练集，A-{Ag}为特征集，递归调用上述步骤
            sub_tree = self.Train(sub_train_df)
            node_tree.Add_node(f, sub_tree)
        return node_tree
    # 方法Train中 ，需要嵌套调用，所以额外用一个Fit方法用来外部调用
    def Fit(self, train_data):
        self._tree = self.Train(train_data)
        return self._tree
    def Predict(self, X_test):
        return self._tree.Predict(X_test)