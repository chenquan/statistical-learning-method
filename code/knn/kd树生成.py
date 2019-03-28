# -*- coding: UTF-8 -*-
# File Name：kd树生成
# Author : Chen Quan
# Date：2019/3/9
# Description :
__author__ = 'Chen Quan'

import numpy as np


class Node:
    def __init__(self, data=None, depth=0, left=None, right=None):
        self.__right = right
        self.__left = left
        self.__data = data
        self.__depth = depth

    @property
    def right(self):
        return self.__right

    @property
    def left(self):
        return self.__left

    @property
    def data(self):
        return self.__data

    @property
    def depth(self):
        return self.__depth

    @right.setter
    def right(self, v):
        self.__right = v

    @left.setter
    def left(self, v):
        self.__left = v

    @data.setter
    def data(self, v):
        self.__data = v

    @depth.setter
    def depth(self, v):
        self.__depth = v


class Tree:
    def __init__(self, data):
        self.tree = None
        self.data = sorted(data, )

        self.create(data)

    def create(self, data, depth=0):
        if len(data) > 0:
            N, k = data.shape
            l = depth % k + 1
            mid = int(N / 2)
            data_sort = sorted(data, key=lambda x: x[l])
            node = Node(data_sort[mid], depth)
            if depth == 1:
                self.tree = node
            node.right = self.create(data_sort[:mid], depth + 1)
            node.left = self.create(data_sort[mid + 1:], depth + 1)


import numpy as np
from math import sqrt
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

iris = load_iris()
df = pd.D(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

data = np.array(df.iloc[:100, [0, 1, -1]])
train, test = train_test_split(data, test_size=0.1)
x0 = np.array([x0 for i, x0 in enumerate(train) if train[i][-1] == 0])
x1 = np.array([x1 for i, x1 in enumerate(train) if train[i][-1] == 1])

tree = Tree(data)
