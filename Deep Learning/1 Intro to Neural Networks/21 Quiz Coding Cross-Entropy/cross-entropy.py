# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:26:47 2019

@author: ruslan.bergenov

https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html
"""


import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    my_cross_entropy = 0.0
    i = 0
    for i in range(len(Y)):
        my_cross_entropy = my_cross_entropy - (Y[i] * np.log(P[i]) + (1 - Y[i]) * np.log(1 - P[i]))

    return my_cross_entropy



myY= [1,1,0]
myP = [0.8,0.7,0.1]

print(cross_entropy(myY, myP))

myY=[0,0,1]

print(cross_entropy(myY, myP))

print(type(cross_entropy(myY, myP)))
