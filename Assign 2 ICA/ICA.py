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
import matplotlib.pyplot as plt

sounds = loadmat('./sounds.mat')['sounds'];

# test = loadmat('./icaTest.mat');
# A = test['A'];
# U = test['U'];

test = loadmat('./sounds.mat');
U = test['sounds'];
#initializing mixing matrix
A = ([0.5, 1, 0.2, 1, 0.3],[0.5, 0.5, 0.2, 0.6, 0.1],[0.8, 0.4, 0.5, 0.5, 0.31],[0.5, 0.4, 0.5, 1, 1],[0.2, 0.7, 0.7, 0.1, 0.3])
print("A=",A)
A=np.array(A)
m = A.shape[0];
n = A.shape[1];

W = ([[0.41113469, 0.21755247, 0.55648479, 0.15306791, 0.84595753],
       [0.34862363, 0.34633332, 0.13030717, 0.80047687, 0.8880103 ],
       [0.23412216, 0.93441739, 0.53875375, 0.79915107, 0.32011519],
       [0.24549826, 0.11758528, 0.25647   , 0.40310727, 0.05423368],
       [0.84298051, 0.12147498, 0.99985129, 0.17120442, 0.6393325 ]]);

W = np.array(W)
#assuming fixed W1, uncomment if W1 is to be random
# W = np.random.rand(m,n);

correl = np.array([0,0,0,0,0]) #initializing correlation matrix

cor_iter = [[0],[0],[0],[0],[0]]

X = np.matmul(A,U)
t = X.shape[1];


eta = 0.01;
dW = np.zeros((n,n))+.01; #dW becomes a shallow copy if dW=W
count = 0;
# W1=W; X1=X;
# ipdb.set_trace();
tol = 1;
tol_limit = 1e-6;
max_iter = 1e5;
# ipdb.set_trace();
#Iterating to calculate unmixing matrix
while (count<max_iter and tol > tol_limit):
    Y = W.dot(X)

    Z = 1/(1+np.exp(-1*Y));
    # ipdb.set_trace();
    dW = eta*((np.eye(n) + (1-2*Z).dot(np.transpose(Y))/t).dot(W))
    W = W + dW
    count = count+1;

    tol = LA.norm(dW)/LA.norm(W)
    # change this order for W.dot(X)[] accordingly if starting value of W is changed
    #retrieving correlation with corresponding retrieved signals
    cor_iter[0] = np.corrcoef(U[0,:],W.dot(X)[2,:])[0,1]
    cor_iter[1] = np.corrcoef(U[1,:],W.dot(X)[4,:])[0,1]
    cor_iter[2] = np.corrcoef(U[2,:],W.dot(X)[1,:])[0,1]
    cor_iter[3] = np.corrcoef(U[3,:],W.dot(X)[0,:])[0,1]
    cor_iter[4] = np.corrcoef(U[4,:],W.dot(X)[3,:])[0,1]
    correl = np.vstack([correl,cor_iter]) # updating correlation matrix with iteration

#retrieving estimate of source signals
Uhat = W.dot(X);
#Playing back audio at 44100/4 Hz
freq = 11025;
sd.play(U[0,:],freq,blocking = True)
sd.play(Uhat[0,:]/27.5,freq, blocking=True)

#Saving data for plotting
data = {"U": U, "What":W, "X":X, "Uhat":Uhat, "correl":correl}
pkl.dump(data,open("plot_data_ICA.p","wb"));
# sd.play(Uhat1[0,:],freq, blocking=True)
ipdb.set_trace();
