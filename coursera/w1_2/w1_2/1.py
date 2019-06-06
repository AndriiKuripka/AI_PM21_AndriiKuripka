import pandas
import numpy as np

from sklearn.tree import DecisionTreeClassifier
data = pandas.read_csv('titanic.csv')

dataMain = data.drop(data.columns[[0,3,6,7,8,10,11]], axis = 1)
dataMain = dataMain.dropna()
survived = dataMain['Survived']
signs = dataMain.drop(dataMain.columns[[0]], axis = 1)
signs = signs.dropna()
signs = signs.replace('female', 0)
signs = signs.replace('male', 1)
signs = signs.dropna()
clf = DecisionTreeClassifier(random_state=241)
clf.fit(signs,survived)
importances = clf.feature_importances_
print(importances[0],"- Pclass")
print(importances[1],"- Fare")
print(importances[2],"- Age")
print(importances[3],"- Sex")