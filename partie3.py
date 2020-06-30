import matplotlib.pyplot as plt
import numpy as np
import random
from math import *
from partie1 import *

def gen_UPPER_BD(n,m,MAX):
    #Function that generate an upper bidiagonal matrix with (n x m) dimension with random number between 0 and MAX
    A=np.zeros((n,m))
    i = 0
    while (i < n-1 and i < m-1):
        A[i][i] = random.randint(0,MAX)
        A[i][i+1] = random.randint(0,MAX)
        i += 1
    A[i][i] = random.randint(0,MAX)
    if (n < m):
        A[i][i+1] = random.randint(0,MAX)
    return A

def gen_LOWER_BD(n,m,MAX):
    #Function that generate a lower bidiagonal matrix with (n x m) dimension with random number between 0 and MAX
    A=np.zeros((n,m))
    A[0][0] = random.randint(0,MAX) 
    i = 1
    while (i < n and i < m):
        A[i][i-1] = random.randint(0,MAX)
        A[i][i] = random.randint(0,MAX)
        i += 1
    if (n > m):
        A[i][i-1] = random.randint(0,MAX)
    return A

def BD_to_D(BD,NMax):
    #Function realizing the SVD Decomposition Using QR Factorizing function from numpy
    #Param : BD a bidiagonal matrix
    #Param : NMax number of iteration
    #Returns U,S,V that verify BD = U*S*V 
    S = BD.copy()
    (n,m)= np.shape(BD)
    U=np.eye(n,n)
    V=np.eye(m,m)
    for i in range(NMax):
        Q1, R1=np.linalg.qr(np.transpose(S))
        Q2, R2=np.linalg.qr(np.transpose(R1))
        S=R2
        U=np.dot(U,Q2)
        V=np.dot(np.transpose(Q1),V)
    return U,S,V

def diagonale(A):
    #Function that returns a matrix with the diagonal of the matrix A 
    n = len(A)
    M=np.zeros((n,n))
    for i in range (n):
        for j in range (n):
            if( i != j):
                M[i][j]=0
            else:
                M[i][j]=A[i][j]
    return M

def conv(BD,D,Nmax):
    #Function that calculate the norm of the matrix (S - D) and S it's resulting from the SVD decomposition
    #Param : BD the bidiagonal matrix
    #Param : D a null matrix with same dimension of BD
    #Param : NMax the number of iteration used for the SVD decomposition
    S = BD_to_D(BD,Nmax)
    D = diagonale(S[1])
    X = S[1]
    X = np.subtract(X,D)
    norme = np.linalg.norm(X)
    return norme

def draw(S,NMax):
    #Function that draws the curve illustrating the convergence of S towards its diagonal matrix
    #Param : NMax is the number maximum of iterations ( supposed to be higher than 50 )
    n=len(S)
    D=np.zeros((n,n))
    A = [i for i in range(5,NMax)]
    B = np.zeros(len(A))
    for i in range(0,len(A)):
        B[i] = conv(S,D,A[i])
    print(B)
    plt.plot(A,B)
    plt.ylabel("||BDn - D ||")
    plt.xlabel("n")
    plt.show()

def Invariant(U,S,V,BD,eps):
    #Function that verify the invariant of the SVD decomposition 
    #Returns 0 if U*S*V = BD , 1 if not
    A = np.dot(U,S)
    A = np.dot(A,V)
    A = np.subtract(A,BD)
    (n,m)=np.shape(BD)
    for i in range (n):
        for j in range(m):
            if (abs(A[i][j]) > eps):
                return False
    return True

def abs_ordred_matrix(A):
    #Function that returns a positive matrix (Absolute value of matrix A) 
    n=len(A)
    B=np.zeros((n,n))
    for i in range (n):
        for j in range(n):
            B[i][j] = abs(A[i][j])
    for i in range(n):
        for j in range(i,n):
            if(B[i][i]<B[j][j]):
                (B[i][i],B[j][j]) = (B[j][j],B[i][i]) 
    return B

def QR_decomposition_iter(A):
    n,m = np.shape(A)
    u = A[:,0]
    v = np.zeros((n,1))
    v[0,0] = np.linalg.norm(u)
    Q = householder(u,v)
    R = np.dot(Q,A)
    for i in range(1,min(n-1,m-1)):
        u = R[i:,i]
        v = np.zeros((len(u),1))
        v[0,0] = np.linalg.norm(u) 
    Qi = np.eye(n)
    Qi[i:,i:] = householder(u,v)
    Q = np.dot(Q,Qi)
    R = np.dot(Qi,R)
    return (Q,R)

def BD_to_D2(BD,NMax):
    #Function realizing the SVD Decomposition Using QR Factorizing function from numpy
    #Param : BD a bidiagonal matrix
    #Param : NMax number of iteration
    #Returns U,S,V that verify BD = U*S*V 
    (n,m)= np.shape(BD)
    U=np.eye(n,n)
    V=np.eye(m,m)
    for i in range(NMax):
        Q1, R1=QR_decomposition_iter(np.transpose(BD))
        Q2, R2=QR_decomposition_iter(np.transpose(R1))
        BD=R2
        U=np.dot(U,Q2)
        V=np.dot(np.transpose(Q1),V)
    return U,BD,V

def svd_rectifie(U,S,V,BD):
    #Function that modify the matrix U and S to assure that the matrix S is positive with a descending order
    n,m = np.shape(S)
    triS = abs_ordred_matrix(S)
    DS = diagonale(S)
    DS /= triS
    for i in range(min(n,m)):
        S[i,i] = triS[i][i]
        U[:,i] = U[:,i] * DS[i][i]
    return U,S,V

def is_diagonal_with_a_decreasing_order(S,epsilon):
    #Function that tests if S is a diagonal decreasing matrix
    n,m = np.shape(S)
    if (n != m):
        print("the matrix given is not a square matrix")
    else:
        for i in range(n):
            for j in range(n):
                if(i != j):
                    if (abs(S[i][j]) > epsilon):
                        return False
    for i in range(n-1):
        if (S[i][i] < S[i][i+1]):
            return False
    return True

def svd_test(n,m,NMax):
    #Function that checks if svd decomposition is working well
    #Param : n,m are the dimension of the matrix 
    #Param : NMax is the number of iteration desired
    BD = gen_LOWER_BD(n,m,100)
    U,S,V = BD_to_D(BD,NMax)
    U,S,V = svd_rectifie(U,S,V,BD)
    return (Invariant(U,S,V,BD,10**(-5)) and is_diagonal_with_a_decreasing_order(S,10**(-5)))

#print(svd_test(6,6,200))

#http://people.inf.ethz.ch/gander/papers/qrneu.pdf
