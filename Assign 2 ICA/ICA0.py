import struct as st
import numpy as np
import keyboard
import idx2numpy
from PIL import Image
from numpy import linalg as LA
from numpy import matlib
import sounddevice as sd
import pickle as pkl
import ipdb;
import ica_functions
from scipy.io import loadmat;
from IPython.display import Audio

sounds = loadmat('./sounds.mat')['sounds'];

test = loadmat('./icaTest.mat');
test = loadmat('./sounds.mat');

# A = test['A'];
# U = test['U'];

U = test['sounds'];
#A = np.array([1, 1, 1, 1, 1],[0.5, 0.5, 0, 1, 1],[0, 0, 0.5, 0.5, 1],[0.5, 0, 0.5, 1, 1],[0, 0.7, 0.7, 0.1, 0.3])
A = ([1, 1, 1, 1, 1],[0.5, 0.5, 0, 1, 1],[0, 0, 0.5, 0.5, 1],[0.5, 0, 0.5, 1, 1],[0, 0.7, 0.7, 0.1, 0.3])
# A = np.array([[0]*5]*5)

print(A)
#print(A[1,:])
#A = np.array([1, 1, 1, 1, 1]);
A=np.array(A)
m = A.shape[0];

#A[1,:]=np.array([0.5, 0.5, 0, 1, 1]) #,[0, 0, 0.5, 0.5, 1],[0.5, 0, 0.5, 1, 1],[0, 0.7, 0.7, 0.1, 0.3])
n = A.shape[1];
W = np.random.rand(m,n);

X = np.matmul(A,U)
t = X.shape[1];

XRowSum = X.sum(axis=1)
print(XRowSum)

#X=X-np.matlib.repmat(XRowSum,t,1).transpose();

eta = 0.001;
dW = np.zeros((n,n))+.01; #dW becomes a shallow copy if dW=W
count = 0;
W1=W; X1=X;
# ipdb.set_trace();
W0,WX,couter = ica_functions.performICA(W1, X1, eta, 1e4, 1e-6);

# while (count<1e4 and LA.norm(dW)/LA.norm(W) > 1e-6):
#     Y = W.dot(X)
#
#     Z = 1/(1+np.exp(-1*Y));
#     # ipdb.set_trace();
#     dW = eta*((np.eye(n) + (1-2*Z).dot(np.transpose(Y))/t).dot(W))
#     W = W + dW
#     count = count+1;
#Audio(X[0,:], rate=11025)


freq = 11025;
sd.play(U[0,:],freq,blocking = True)

# Uhat = W.dot(X);
Uhat1 = W0.dot(X)
# sd.play(Uhat[0,:],freq, blocking=True)
sd.play(Uhat1[0,:],freq, blocking=True)
ipdb.set_trace();
#Audio(X, rate=11025)
