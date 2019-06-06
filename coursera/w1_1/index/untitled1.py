# -*- coding: utf-8 -*-
"""
Created on Mon May 20 21:51:03 2019

@author: andre
"""
import pandas
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
import numpy as np
from sklearn.tree import DecisionTreeClassifier
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
clf = DecisionTreeClassifier()
clf.fit(X, y)