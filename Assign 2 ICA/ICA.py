import struct as st
import numpy as np
import keyboard
import idx2numpy
from PIL import Image
from numpy import linalg as LA
import pickle as pkl

from scipy.io import loadmat;

sounds = loadmat('./sounds.mat')['sounds'];

test = loadmat('./icaTest.mat');
#test = loadmat('./sounds.mat');

A = test['A'];
U = test['U'];

m = A.shape[0];
n = A.shape[1];
W = np.eye(3,3)+2;#np.random.rand(m,n)+1;

X = np.matmul(A,U)

Y = np.matmul(W,X)

Z = 1/(1+np.exp(-Y));

eta = 0.0001;
dW = W;
count = 0;

while LA.norm(dW)/LA.norm(W) > 0.005:
    dW = eta*(np.eye(n,n) + np.matmul((1-2*Z),Y.transpose()))*W
    W = W + dW
    count = count+1;

Uhat = np.matmul(W,X);
