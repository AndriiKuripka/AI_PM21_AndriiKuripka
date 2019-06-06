# -*- coding: utf-8 -*-
"""
@author: andre
"""
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append("..")
'''loading samples of learning and testing'''
df_train = pd.read_csv("perceptron-train.csv", header=None)
X_train = df_train.loc[:, 1:]
y_train = df_train[0]
df_test = pd.read_csv("perceptron-test.csv", header=None)
X_test = df_test.loc[:, 1:]
y_test = df_test[0]
'''learning perceptron'''
model = Perceptron(max_iter=5, tol=None, random_state=241)
model.fit(X_train, y_train)
'''calculate quality of classifier on test sample'''
acc_before = accuracy_score(y_test, model.predict(X_test))
acc_before
'''normalize the sample'''
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
'''learning on the new sample'''
model.fit(X_train_scaled, y_train)
acc_after = accuracy_score(y_test, model.predict(X_test_scaled))
acc_after
diff = acc_after - acc_before
print(1, f"{diff:.3f}")