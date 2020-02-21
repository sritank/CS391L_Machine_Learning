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
A = ([0.5, 1, 0.2, 1, 0.3],[0.5, 0.5, 0.2, 0.6, 0.1],[0.8, 0.4, 0.5, 0.5, 0.31],[0.5, 0.4, 0.5, 1, 1],[0.2, 0.7, 0.7, 0.1, 0.3])
# A = np.array([[0]*5]*5)

print(A)
#print(A[1,:])
#A = np.array([1, 1, 1, 1, 1]);
A=np.array(A)
m = A.shape[0];

#A[1,:]=np.array([0.5, 0.5, 0, 1, 1]) #,[0, 0, 0.5, 0.5, 1],[0.5, 0, 0.5, 1, 1],[0, 0.7, 0.7, 0.1, 0.3])
n = A.shape[1];
W = np.random.rand(m,n);
# W = [[0.79722847, 0.74092872, 0.96148456, 0.33723867, 0.42588017],
#        [0.2934053 , 0.82959767, 0.88318573, 0.41958464, 0.63933106],
#        [0.57606894, 0.76955245, 0.07654651, 0.64246145, 0.90654973],
#        [0.2412909 , 0.10643112, 0.4064552 , 0.32060839, 0.76532299],
#        [0.44236227, 0.02811083, 0.70773259, 0.39954014, 0.8544893 ]];
# W = np.array(W)

X = np.matmul(A,U)
t = X.shape[1];

XRowSum = X.sum(axis=1)
print(XRowSum)

#X=X-np.matlib.repmat(XRowSum,t,1).transpose();

eta = 0.002;
dW = np.zeros((n,n))+.01; #dW becomes a shallow copy if dW=W
count = 0;
W1=W; X1=X;
# ipdb.set_trace();
# W0,WX,couter = ica_functions.performICA(W1, X1, eta, 1e4, 1e-4);
tol = 1;
tol_limit = 1e-6;
max_iter = 1e5;
while (count<max_iter and tol > tol_limit):
    Y = W.dot(X)

    Z = 1/(1+np.exp(-1*Y));
    # ipdb.set_trace();
    dW = eta*((np.eye(n) + (1-2*Z).dot(np.transpose(Y))/t).dot(W))
    W = W + dW
    count = count+1;
    tol = LA.norm(dW)/LA.norm(W)
#Audio(X[0,:], rate=11025)


freq = 11025;
sd.play(U[0,:],freq,blocking = True)

Uhat = W.dot(X);
# Uhat1 = W0.dot(X)
sd.play(Uhat[0,:],freq, blocking=True)
# sd.play(Uhat1[0,:],freq, blocking=True)
ipdb.set_trace();
#Audio(X, rate=11025)
