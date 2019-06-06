# -*- coding: utf-8 -*-
"""
@author: andre
"""
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from typing import Tuple

import sys
sys.path.append("..")
''' uploading the sample '''
data = load_boston()
X = data.data
y = data.target
'''scaling data of the sample'''
X = scale(X)
'''different variant of metrics p=[1,10]. 200 variants of р'''
cv = KFold(n_splits=5, shuffle=True, random_state=42)
def get_best_score(X: np.array, y: np.array, cv) -> Tuple[float, float]:
    best_score, best_p = None, None

    for p in np.linspace(1, 10, 200):
        model = KNeighborsRegressor(p=p, n_neighbors=5, weights="distance")
        score = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error").mean()

        if best_score is None or score > best_score:
            best_score, best_p = score, p
    
    return best_score, best_p

score, p = get_best_score(X, y, cv)

print(1,':', f"{p:.2f}")
