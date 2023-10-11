# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:11:21 2022

@author: rm19adr
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def heading(s):
    """ Return a string with s between two rows of lines.
    """
    line = '-'*len(s)
    return '\n'.join([line,s,line])

print(heading('Exercise 5'))

# method 1
print('Using pandas:')
data = pd.read_csv('data2.csv', sep=' ')
print('data has the following type:', type(data))
X = np.array(data.iloc[:,0])
Y = np.array(data.iloc[:,1])
print('X=', X)
print('Y=', Y)

# method 2
print('Using numpy:')
data = np.genfromtxt('data2.csv', delimiter=' ')
print('data has the following type:', type(data))
X = data[:,0]
Y = data[:,1]
print('X=', X)
print('Y=', Y)

print(heading('Exercise 6'))
plt.figure(figsize=(8,6))
plt.plot(X,Y)
plt.show()