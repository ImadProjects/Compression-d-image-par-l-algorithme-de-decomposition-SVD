#!/usr/bin/python3.4

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import math as m
import os
from partie1 import *

from matplotlib.pyplot import figure, show


M = np.matrix([[11,21,5,12],[48,548,16,887],[47,88,91,45],[17,32,68,17]])


def extraireVect_Col(M,n,m,i):
    #Function that extract the column vector with index i from the matrix M
    Y = np.zeros(shape=(n,1))
    for j in range(i,n): #i
        Y[j] = M[j,i]
    return Y

def extraireVect_Lig(M,n,m,i):
    #Function that extract the line vector with index i from the matrix M
    Y = np.zeros(shape=(m,1))
    for j in range(i+1,m): #i+1
        Y[j] = M[i,j]
    return Y

def genererVect(X):
    Y = np.zeros(shape=np.shape(X))
    i = 0
    while (i < np.shape(X)[0] and X[i] == 0):
        i += 1
    if (i < np.shape(X)[0]):
        Y[i] = np.linalg.norm(X)
    return Y


def BD(A):
    #Function that transforms the matrix A to a bidiagonal matrix BD and two others Qleft and Qright
    #The matrix provide A = Qleft x BD x Qright
    n,m = np.shape(A)
    Qleft = np.eye(n)
    Qright = np.eye(m)
    BD = A
    for i in range(n-1):
        X = extraireVect_Col(BD,n,m,i)
        Y = genererVect(X)
        if (np.linalg.norm(X) == 0 and np.linalg.norm(Y) == 0):
            Q1 = np.eye(X.size)
        else:
            Q1 = householder(X,Y)
        Qleft = np.dot(Qleft, Q1)
        BD = np.dot(Q1, BD)
        if (i < (m-2)):
            X2 = extraireVect_Lig(BD,n,m,i)
            Y2 = genererVect(X2)
            if (np.linalg.norm(X2) == 0 and np.linalg.norm(Y2) == 0):
                Q2 = np.eye(X2.size)
            else:
                Q2 = householder(X2,Y2)
            Qright = np.dot(Q2, Qright)
            BD = np.dot(BD, Q2)
    return (Qleft, BD, Qright)
'''
Qleft, BD2, Qright = BD(M)

A = np.dot(np.dot(Qleft,BD2), Qright)


ec_rel = np.linalg.norm(A-M)/np.linalg.norm(M)

print(A)

print(ec_rel)
'''
