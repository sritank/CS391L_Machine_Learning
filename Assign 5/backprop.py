import csv
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
from scipy.io import loadmat;
from IPython.display import Audio
import matplotlib.pyplot as plt
import random
import sys
alpha=1;
def actFcn(u):
    return (1/(1+np.exp(-alpha*u)))


def actFcnDer(u):
    return (alpha*np.exp(-alpha*u)/(1+np.exp(-alpha*u))**2)

def feedForward(u1,W2,W3):
    act2 = actFcn(W2.dot(u1)).reshape(-1,1);
    # ipdb.set_trace();
    act2_b = np.vstack([np.ones((1,act2.shape[1])), act2])
    return(actFcn(W3.dot(act2_b)));

#***********Reading training and test data*********************
train_img = idx2numpy.convert_from_file('train-images-idx3-ubyte');
test_img = idx2numpy.convert_from_file('t10k-images-idx3-ubyte');
train_lab = idx2numpy.convert_from_file('train-labels-idx1-ubyte');
test_lab = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte');
#**************************************************************
L_train = train_img.shape[0];
L_test = test_img.shape[0];
train_img = train_img.reshape(L_train,28*28)/255;
test_img = test_img.reshape(L_test,28*28)/255;

n2 = 15;
n1 = train_img.shape[1];
n3=10;
W2_b = np.random.rand(n2, n1+1)*1e-2;
W2 = W2_b[:,1:];

#******Adding bias term to input data*******************
train_img_input = np.vstack([np.ones((1,L_train)), train_img.T]);
test_img_input = np.vstack([np.ones((1,L_test)), test_img.T]);
#******calculating activations of layer 2 = hidden layer*******************
# z2 = W2_b.dot(train_img_input);
# a2 = actFcn(z2);#outputs of layer 2 for all input images
#*****************Activations of layer 3 = output layer*******************
#
# a2_b = np.vstack([np.ones((1,L_train)), a2]);
# W3_b = np.random.rand(n3, n2+1)*1e-2;
# W3 = W3_b[:,1:];
# z3 = W3_b.dot(a2_b);
# a3 = actFcn(z3);

#***********converting digit label to vector label, i.e 1=[0,1,0,...0]', 5=[0,0,0,0,0,1,0,0,0,0]', etc.
train_lab = train_lab.reshape(-1,1)
train_lab_vec = np.zeros(L_train*n3);
cols = np.arange(0,L_train,1);
cols = cols.reshape(-1,1);
train_lab_vec[10*cols + train_lab]=1;
train_lab_vec = (train_lab_vec.reshape(L_train,n3)).transpose();
#****************************************************************************************

eta=5e-2;
tol = 6e-3;
err_cum = tol*1e3;
err_cum_prev=err_cum*1e3;
# train_samples = L_train;
train_samples = 60000;
test_samples = 10000;


# for j in range(0,1000):
# while err_cum>tol:# and err_cum_prev>err_cum:
#
#     Delta2 = W2_b*0.0;
#     Delta3 = W3_b*0.0;
#     # err_cum_prev=err_cum;
#     err_cum = 0.0;
#     # for i in range(0,train_samples):
#
#     z2 = W2_b.dot(train_img_input[:,:train_samples]);
#     z2_b = np.vstack([np.ones((1,z2.shape[1])), z2]);
#     a2 = actFcn(z2);
#     a2_b = np.vstack([np.ones((1,a2.shape[1])), a2]);
#
#     z3 = W3_b.dot(a2_b);
#     a3 = actFcn(z3);
#
#     # delta3 = np.multiply(a3 - train_lab_vec[:,i].reshape(-1,1),actFcnDer(z3))#if using RMS cost function
#     delta3 = -(train_lab_vec[:,:train_samples] - a3);#if using log cost function from Andrew Ng
#     # delta3 = np.multiply(-(train_lab_vec[:,i].reshape(-1,1) - a3), actFcnDer(z3));
#     # delta2 = np.multiply(W3_b.T.dot(delta3), a2_b)
#     delta2 = np.multiply(W3_b.T.dot(delta3), actFcnDer(z2_b));
#     delta2 = np.delete(delta2,0,0);#delete entire first row
#
#     # o=feedForward(train_img_input[:,0],W2_b,W3_b);
#
#
#     # d3 = delta3[:,i].reshape(-1,1);
#     # d3 = delta3.reshape(-1,1);
#     d3 = delta3.reshape(n3,1,train_samples)
#     # ipdb.set_trace()
#     # a2_b_vec = a2_b[:,i].reshape(-1,1);
#     # a2_b_vec = a2_b.reshape(-1,1);
#     a2_b_vec = a2_b.reshape(n2+1,1,train_samples);
#     a2_b_vec = a2_b_vec.reshape(1,n2+1,train_samples);
#     # Delta3 = Delta3 + d3.dot(a2_b_vec.T)/L_train;
#     Delta3 = Delta3 + d3.dot(a2_b_vec.T)/train_samples;#if using log cost function
#     # ipdb.set_trace()
#     # d2 = delta2[:,i].reshape(-1,1);
#     # d2 = delta2.reshape(-1,1);
#     d2 = delta2.reshape(n2,1,train_samples);
#     ipdb.set_trace()
#     # a1_b_vec = train_img_input[:,:train_samples].reshape(-1,1);
#     a1_b_vec = train_img_input[:,:train_samples].reshape(n1+1,1,train_samples);
#     # Delta2 = Delta2 + d2.dot(a1_b_vec.T)/L_train;
#     Delta2 = Delta2 + d2.dot(a1_b_vec.T)/train_samples;#if using log cost function
#     err_cum = err_cum + np.mean(np.multiply(delta3,delta3));
#         # err_cum = err_cum + np.mean(np.multiply((train_lab_vec[:,i].reshape(-1,1) - a3),(train_lab_vec[:,i].reshape(-1,1) - a3)));#if using RMS cost function
#     err_cum = err_cum/train_samples;
#     print(err_cum)
#     W3_b = W3_b - eta*Delta3;
#     W2_b = W2_b - eta*Delta2;
    # ipdb.set_trace()


#********************Using Cumulative gradient descent******************************
# while err_cum>tol:# and err_cum_prev>err_cum:
#     Delta2 = W2_b*0.0;
#     Delta3 = W3_b*0.0;
#     # err_cum_prev=err_cum;
#     err_cum = 0.0;
#     for i in range(0,train_samples):
#
#         z2 = W2_b.dot(train_img_input[:,i].reshape(-1,1));
#         z2_b = np.vstack([np.ones((1,z2.shape[1])), z2]);
#         a2 = actFcn(z2);
#         a2_b = np.vstack([np.ones((1,a2.shape[1])), a2]);
#
#         z3 = W3_b.dot(a2_b);
#         a3 = actFcn(z3);
#
#         # delta3 = np.multiply(a3 - train_lab_vec[:,i].reshape(-1,1),actFcnDer(z3))#if using RMS cost function
#         delta3 = -(train_lab_vec[:,i].reshape(-1,1) - a3);#if using log cost function from Andrew Ng
#         # delta3 = np.multiply(-(train_lab_vec[:,i].reshape(-1,1) - a3), actFcnDer(z3));
#         # delta2 = np.multiply(W3_b.T.dot(delta3), a2_b)
#         delta2 = np.multiply(W3_b.T.dot(delta3), actFcnDer(z2_b));
#         delta2 = np.delete(delta2,0,0);
#
#         # o=feedForward(train_img_input[:,0],W2_b,W3_b);
#
#
#         # d3 = delta3[:,i].reshape(-1,1);
#         d3 = delta3.reshape(-1,1);
#         # ipdb.set_trace()
#         # a2_b_vec = a2_b[:,i].reshape(-1,1);
#         a2_b_vec = a2_b.reshape(-1,1);
#         # Delta3 = Delta3 + d3.dot(a2_b_vec.T)/L_train;
#         Delta3 = Delta3 + d3.dot(a2_b_vec.T)/train_samples;#if using log cost function
#         # ipdb.set_trace()
#         # d2 = delta2[:,i].reshape(-1,1);
#         d2 = delta2.reshape(-1,1);
#         a1_b_vec = train_img_input[:,i].reshape(-1,1);
#         # Delta2 = Delta2 + d2.dot(a1_b_vec.T)/L_train;
#         Delta2 = Delta2 + d2.dot(a1_b_vec.T)/train_samples;#if using log cost function
#         err_cum = err_cum + np.mean(np.multiply(delta3,delta3));
#         # err_cum = err_cum + np.mean(np.multiply((train_lab_vec[:,i].reshape(-1,1) - a3),(train_lab_vec[:,i].reshape(-1,1) - a3)));#if using RMS cost function
#     err_cum = err_cum/train_samples;
#     print(err_cum)
#     W3_b = W3_b - eta*Delta3;
#     W2_b = W2_b - eta*Delta2;
#     # ipdb.set_trace()
#********************************************************************************



#********************Using Stochastic gradient descent******************************
while err_cum>tol:# and err_cum_prev>err_cum:

    # err_cum_prev=err_cum;
    err_cum = 0.0;
    for i in range(0,train_samples):
        Delta2 = W2_b*0.0;
        Delta3 = W3_b*0.0;
        z2 = W2_b.dot(train_img_input[:,i].reshape(-1,1));
        z2_b = np.vstack([np.ones((1,z2.shape[1])), z2]);
        a2 = actFcn(z2);
        a2_b = np.vstack([np.ones((1,a2.shape[1])), a2]);

        z3 = W3_b.dot(a2_b);
        a3 = actFcn(z3);

        delta3 = np.multiply(a3 - train_lab_vec[:,i].reshape(-1,1),actFcnDer(z3))#if using RMS cost function
        # delta3 = -(train_lab_vec[:,i].reshape(-1,1) - a3);#if using log cost function from Andrew Ng

        delta2 = np.multiply(W3_b.T.dot(delta3), actFcnDer(z2_b));
        delta2 = np.delete(delta2,0,0);

        d3 = delta3.reshape(-1,1);
        # ipdb.set_trace()
        a2_b_vec = a2_b.reshape(-1,1);
        Delta3 = Delta3 + d3.dot(a2_b_vec.T);
        # ipdb.set_trace()

        d2 = delta2.reshape(-1,1);
        a1_b_vec = train_img_input[:,i].reshape(-1,1);
        # Delta2 = Delta2 + d2.dot(a1_b_vec.T)/L_train;
        Delta2 = Delta2 + d2.dot(a1_b_vec.T);#if using log cost function
        # err_cum = err_cum + np.mean(np.multiply(delta3,delta3));
        W3_b = W3_b - eta*Delta3;
        W2_b = W2_b - eta*Delta2;
        err_cum = err_cum + np.mean(np.multiply((train_lab_vec[:,i].reshape(-1,1) - a3),(train_lab_vec[:,i].reshape(-1,1) - a3)));#if using RMS cost function
    err_cum = err_cum/train_samples;
    print(err_cum)
#**************************************************************************************************************



# test_image_no = 1005
# o=feedForward(train_img_input[:,test_image_no],W2_b,W3_b);
# o_label = np.argmax(o);
# print(o)
# print(o_label)
# print(train_lab_vec[:,test_image_no])
# print(train_lab[test_image_no])

correct_pred_test=0;

for i in range(0,test_samples):
    predicted_label = np.argmax(feedForward(test_img_input[:,i],W2_b,W3_b));
    actual_label = test_lab[i];
    if actual_label==predicted_label:
        correct_pred_test = correct_pred_test + 1;



print("pred acc",correct_pred_test)
ipdb.set_trace()
(Image.fromarray(train_img[test_image_no,:].reshape(28,28))).show()
