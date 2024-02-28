# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:50:11 2022

@author: user
"""

from matplotlib import image as img
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas

X = np.array(pandas.read_csv("problem1data.csv", delimiter=' '))

plt.scatter(X[:,0], X[:,1], marker='x', s=50)
plt.show()


km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
lbls = km.fit_predict(X)

plt.scatter(X[:,0], X[:,1], marker='o', c=lbls, s=50)
plt.show()