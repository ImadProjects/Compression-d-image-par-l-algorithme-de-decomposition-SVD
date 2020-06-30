import numpy as np 
import random as rand 
from math import sqrt
import time

def householder ( x , y ):
    #Function given x and y two vectors return the householder matrix H such that H.x=y
    n=np.size(x)
    m=np.size(y)
    #eleminating the cases where the householder matrix can not be generated  
    if (n!=m or (np.array_equal(x,np.zeros((n,1))) and (not np.array_equal(y,np.zeros((n,1)))))):
        print  ("erreur")
        exit(0)
    U= (x-y)/np.linalg.norm(x-y)
    n=x.size
    return np.eye(n,n)-2*np.dot(U,U.T)
def householder2 ( x , y ):
    #anothoer method :Function given x and y two vectors return the householder matrix H such that H.x=y
    n=np.size(x)
    m=np.size(y)
    #eleminating the cases where the householder matrix can not be generated  
    if (n!=m or (np.array_equal(x,np.zeros((n,1))) and (not np.array_equal(y,np.zeros((n,1)))))):
        print  ("erreur")
        exit(0)
    c=0
    for i in range(0,n):
        c+=x[i,0]*x[i,0]-x[i,0]*y[i,0]
    c/=2.0
    c=abs(c)**(1/2)
    U=np.zeros((n,1))
    TU=np.zeros((1,n))
    I=np.eye(n)
    for i in range(0,n):
        U[i,0]=(1/(2.0*c))*(x[i,0]-y[i,0])
        TU[0,i]=U[i,0]
    return (I-2*np.dot(U,TU))


def test_holder():
    #tests returns true if the matrix returned by generating householder matrix from two random vectors x,y verifies H.x=y and false otherwise 
    n=rand.randint(4,10)
    x=np.zeros((n,1))
    y=np.zeros((n,1))
    for i in range (0,n):
        x[i,0]=rand.randint(-100,100)
        y[i,0]=rand.randint(-100,100)
    nx=np.linalg.norm(x)
    ny=np.linalg.norm(y)
    A=householder((1/nx)*x,(1/ny)*y)
    return np.allclose(np.dot(A,x)*(ny/nx),y)
#print(test_holder())
def product_MV(H,X):
    #function given H a householder matrix and and X a vecter return H*x with  a complixity lower than numpy.dot (n**2)
    n=np.shape(H)[0]
    A=(H-np.identity(n))*-0.5
    Y=np.zeros((n,1))
    L=np.zeros((n,1))
    L[0,0]=sqrt(A[0][0])
    for j in range(1,n):
        if A[0][j]>=0:
            L[j,0]=sqrt(A[j][j])
        else:
            L[j,0]=-sqrt(A[j][j])
    t=np.dot(L.T,X)
    return X-2*np.dot(L,t)

def product_MM(H,M):
    #function given H a householder matrix and M a matrix within the same size returns H*M with a comlexity lower than numpy.dot ( n**3)
    n=np.shape(H)[0]
    A=np.zeros((n,n))
    for i in range(0,n):
        A[:,i]=product_MV(H,M[:,i]).reshape(n)
    return A

def prod_test():
    #test returns true if the multiplication of a householder with a vector or a matrix return the correct answer with a better complixity than numpy.dot (n**3)  and false otherwise
    n=3
    x=np.random.randint(100,size=(n,1))
    y=np.random.randint(100,size=(n,1))
    z=np.random.randint(100,size=(n,n))
    nx=np.linalg.norm(x)
    ny=np.linalg.norm(y)
    A=householder((1/nx)*x,(1/ny)*y)
    return np.allclose(product_MM(A,z),np.dot(A,z))
    
#print(prod_test())
