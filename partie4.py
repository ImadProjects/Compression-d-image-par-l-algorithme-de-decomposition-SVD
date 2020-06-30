import random as rand
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import os 
from partie3 import *

def decomp_RBG(A):
    #Function given a matrix in shape of (n,n,3)returns 3 matrixs od its components
    #param:A a (n,n,3) size matrix
    #returns: R,B,G 3(n,n) size matrixs 
    n=np.size(A,1)
    R=np.zeros((n,n))
    B=np.zeros((n,n))
    G=np.zeros((n,n))
    for i in range (0,n):
        for j in range (0,n):
            R[i][j]=A[i][j][0]
            B[i][j]=A[i][j][1]
            G[i][j]=A[i][j][2]
    return R,B,G


def comp_RBG(R,B,G):
    #Function given 3 matrixs  with the size (n,n) returns 1 matrix with the size (n,n,3) that containes the 3 matrixs 
    n=np.size(R,1)
    A=np.zeros((n,n,3))
    for i in range (0,n):
        for j in range (0,n):
            #making sure  that the RBG conponents belongs strictly to [0,1]
            if (R[i][j]<1 and R[i][j]>0) :
                
                A[i][j][0]=R[i][j]
            elif R[i][j]>1 : 
                A[i][j][0]=1
            else :
                A[i][j][0]=0
            if (B[i][j]<1 and B[i][j]>0) :
                A[i][j][1]=B[i][j]
            elif B[i][j]>1 : 
                A[i][j][1]=1
            else :
                A[i][j][1]=0
            if (G[i][j]<1 and G[i][j]>0):
               A[i][j][2]=G[i][j]
            elif G[i][j]>1 :
                A[i][j][2]=1
            else :
                A[i][j][2]=0
           
    return A


def compress_k_RBG (A,k):
    #Function that given a matrix with a size of (n,n)  A and an integer k returns a compressed matrix  to the rank k using SVD decomposition 
    u,s,v=np.linalg.svd(A)
    n=np.size(A,1)
    for i in range (k,n):
        s[i]=0
    return np.dot(u,np.dot(np.diag(s),v))



def compress_k(A,k):
    #Function that given  a matrix with size of (n,n,3) and an integer K return a compressed matrix to the rank K using svd 
    R,B,G=decomp_RBG(A)
    R=compress_k_RBG(R,k)
    B=compress_k_RBG(B,k)
    G=compress_k_RBG(G,k)
    return comp_RBG(R,B,G)


def test_compress(fic) :
    #Test that given a png file show this file compressed to the rank 10 , 50 
    img = mp.image.imread(fic)
    plt.subplot(131)
    plt.imshow(img)
    t=plt.title("image original")
    t.set_fontsize(8)
    plt.subplot(132)
    img2=compress_k(img,50)
    plt.imshow(img2)
    t=plt.title(" rang 50")
    t.set_fontsize(8)
    plt.subplot(133)
    img2=compress_k(img,10)
    plt.imshow(img2)
    t=plt.title(" rang 10")
    t.set_fontsize(8)
    plt.show()
#test_compress("pp3.png")
test_compress("p3.png")
def error (ima,k):
    #Function that given a png file and a rank k  compress the file to k rank  and calculate the distance between the the png file and the compressed image  for each color and the then the total distance 
    initial_image=mp.image.imread(ima)
    Rinit,Binit,Ginit=decomp_RBG(initial_image)
    Rcomp=compress_k_RBG(Rinit,k)
    Bcomp=compress_k_RBG(Binit,k)
    Gcomp=compress_k_RBG(Ginit,k)
    ER=np.linalg.norm(Rcomp-Rinit)
    EB=np.linalg.norm(Bcomp-Binit)
    EG=np.linalg.norm(Gcomp-Ginit)
    ET=np.linalg.norm(comp_RBG(Rcomp,Bcomp,Gcomp)-initial_image)
    return ER,EB,EG,ET
def relative_error(ima,k):
    #Function that given a png file and a rank k  compress the file to k rank  and calculate the distance between the the png file and the compressed version divided by the norm of the initial matrix for each color and the then the total relative error 
    initial_image=mp.image.imread(ima)
    Rinit,Binit,Ginit=decomp_RBG(initial_image)
    ER,EB,EG,ET=error(ima,k)
    return ER/np.linalg.norm(Rinit),EB/np.linalg.norm(Binit),EG/np.linalg.norm(Ginit),ET/np.linalg.norm(initial_image)


def plot_error (img,f,f_name):
    #Function that givien a png file and a function that returns 4 flots plot 4 graphs represebting that function on the interval( 0, 500)
    x=np.arange(0,500,25,dtype=int)
    r=np.linspace(0,0,20)
    b=np.linspace(0,0,20)
    g=np.linspace(0,0,20)
    t=np.linspace(0,0,20)
    for i in range (0,20):
        r[i],b[i],g[i],t[i]=f(img,x[i])
    plt.plot(x,r,'r--',label=f_name+" for red")
    plt.plot(x,b,'b--',label=f_name+" for bleu")
    plt.plot(x,g,'g--',label=f_name+" for green" )
    plt.plot(x,t,'k',label="total "+f_name)
    plt.ylabel(f_name)
    plt.xlabel('compression to the rank k')
    plt.title(f_name+" for different rank compression" )
    plt.legend()
    plt.show()
#plot_error("p3.png",error,"error")
#plot_error("p3.png",relative_error,"relative error")   
    
def image_storage(ima,k):
    #Function given a png file and an integer K return the compressed to the rank k file size    
    image=mp.image.imread(ima)
    image_compressed=compress_k(image,k)
    plt.imsave('compressed_to_delete.png',image_compressed)
    return (os.path.getsize('compressed_to_delete.png'))
def plot_storage(ima):
    #function that given a png file plot the grah showing the evolution of the compressed file to the rank k as a  function of k  
    X=np.arange(0,300,10)
    storage=[image_storage(ima,i) for i in X ]
    plt.plot(X,storage)
    plt.show()
#plot_storage("p3.png")
