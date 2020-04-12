

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
from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize
from numpy.linalg import inv


def kernel(X1, X2, l=1.0, sigma_f=1.0):
    '''
    Isotropic squared exponential kernel. Computes
    a covariance matrix from points in X1 and X2.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        Covariance matrix (m x n).
    '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    '''
    Computes the suffifient statistics of the GP posterior predictive distribution
    from m training data X_train and Y_train and n new inputs X_s.

    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.

    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    '''
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = LA.inv(K)

    # Equation (4)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (5)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s



def nll_fn(X_train, Y_train, naive=True):
    '''
    Returns a function that computes the negative log marginal
    likelihood for training data X_train and Y_train and given
    noise level.

    Args:
        X_train: training locations (m x d).
        Y_train: training targets (m x 1).
        noise: known noise level of Y_train.
        naive: if True use a naive implementation of Eq. (7), if
               False use a numerically more stable implementation.

    Returns:
        Minimization objective.
    '''
    def nll_naive(theta):
        # Naive implementation of Eq. (7). Works well for the examples
        # in this article but is numerically less stable compared to
        # the implementation in nll_stable below.
        noise = theta[2]
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise**2 * np.eye(len(X_train))
        return 0.5 * np.log(det(K)) + \
               0.5 * Y_train.T.dot(inv(K).dot(Y_train)) + \
               0.5 * len(X_train) * np.log(2*np.pi)

    def nll_stable(theta):
        # Numerically more stable implementation of Eq. (7) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.
        noise = theta[2]
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise**2 * np.eye(len(X_train))
        L = cholesky(K)
        return np.sum(np.log(np.diagonal(L))) + \
               0.5 * Y_train.T.dot(lstsq(L.T, lstsq(L, Y_train)[0])[0]) + \
               0.5 * len(X_train) * np.log(2*np.pi)

    if naive:
        return nll_naive
    else:
        return nll_stable






marker = '15';
marker_x = marker+'_x';
marker_y = marker+'_y';
marker_z = marker+'_z';
marker_c = marker+'_c';
x_t = 'frame'

#***************Reading data from file********************************************



sigma_l_arr = 0;
sigma_f_arr = 0;
sigma_n_arr = 0;

#********************** Obtaining the data *********************************
counter = 0;
data1 = np.array([0 , 0, 0]);
data2 = np.array([0 , 0, 0]);
data3 = np.array([0 , 0, 0]);
data4 = np.array([0 , 0, 0]);
data5 = np.array([0 , 0, 0]);
with open('./data_GP/AG/block1-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213203046-59968-right-speed_0.500.csv', newline='') as csvfile:
     reader = csv.DictReader(csvfile)
     for row in reader:
         counter =counter+1;
         # print(row['frame'], row['0_x'])
         # ipdb.set_trace();
         # if(counter>=frame_start and counter<frame_start+window_size):
         data1 = np.vstack([data1, np.array([row[x_t], row[marker_x], row[marker_c]],dtype=float)]);

data1=np.delete(data1,0,0);

counter=0
with open('./data_GP/AG/block2-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204004-59968-right-speed_0.500.csv', newline='') as csvfile:
     reader = csv.DictReader(csvfile)
     for row in reader:
         counter =counter+1;
         # print(row['frame'], row['0_x'])
         # if(counter>=frame_start and counter<frame_start+window_size):
         data2 = np.vstack([data2, np.array([row[x_t], row[marker_x], row[marker_c]],dtype=float)]);

data2=np.delete(data2,0,0);

counter=0
with open('./data_GP/AG/block3-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204208-59968-right-speed_0.500.csv', newline='') as csvfile:
     reader = csv.DictReader(csvfile)
     for row in reader:
         counter =counter+1;
         # print(row['frame'], row['0_x'])
         # if(counter>=frame_start and counter<frame_start+window_size):
         data3 = np.vstack([data3, np.array([row[x_t], row[marker_x], row[marker_c]],dtype=float)]);
data3=np.delete(data3,0,0);
#
#
#
counter=0
with open('./data_GP/AG/block4-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204925-59968-right-speed_0.500.csv', newline='') as csvfile:
     reader = csv.DictReader(csvfile)
     for row in reader:
         counter =counter+1;
         # print(row['frame'], row['0_x'])
         # if(counter>=frame_start and counter<frame_start+window_size):
         data4 = np.vstack([data4, np.array([row[x_t], row[marker_x], row[marker_c]],dtype=float)]);
data4=np.delete(data4,0,0);
#
#
counter=0
with open('./data_GP/AG/block5-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213210121-59968-right-speed_0.500.csv', newline='') as csvfile:
     reader = csv.DictReader(csvfile)
     for row in reader:
         counter =counter+1;
         # print(row['frame'], row['0_x'])
         # if(counter>=frame_start and counter<frame_start+window_size):
         data5 = np.vstack([data5, np.array([row[x_t], row[marker_x], row[marker_c]],dtype=float)]);
data5=np.delete(data5,0,0);
data = np.zeros((1010,2));

# ************** mixing 5 traces **********************************
counter=0;
flag=0;
for i in range(0,1010):
    r=random.randint(1,5);
    flag=0;
    while flag==0:
        if data1[i,2]<0 and data2[i,2]<0 and data3[i,2]<0 and data4[i,2]<0 and data5[i,2]<0:
            flag=1;
            continue;
        elif r==1 and data1[i,2]>0:
            data[counter,:] = data1[i,:-1];
            counter=counter+1;
            flag=1;
        elif r==2 and data2[i,2]>0:
            data[counter,:] = data2[i,:-1];
            counter=counter+1;
            flag=1;
        elif r==3 and data3[i,2]>0:
            data[counter,:] = data3[i,:-1];
            counter=counter+1;
            flag=1;
        elif r==4 and data4[i,2]>0:
            data[counter,:] = data4[i,:-1];
            counter=counter+1;
            flag=1;
        elif r==5 and data5[i,2]>0:
            data[counter,:] = data5[i,:-1];
            counter=counter+1;
            flag=1;
        else:
            r=random.randint(1,5);

# ************** mixing 2 traces **********************************
counter=0;
flag=0;
for i in range(0,1010):
    r=random.randint(1,2);
    flag=0;
    while flag==0:
        if data1[i,2]<0 and data2[i,2]<0:
            flag=1;
            continue;
        elif r==1 and data1[i,2]>0:
            data[counter,:] = data1[i,:-1];
            counter=counter+1;
            flag=1;
        elif r==2 and data2[i,2]>0:
            data[counter,:] = data2[i,:-1];
            counter=counter+1;
            flag=1;
        else:
            r=random.randint(1,2);


counter=0;
# for i in range(0,1010):
#     if data1[i,2]<=0:
#         continue;
#     else:
#         data[counter,:] = data1[i,:-1];
#         counter=counter+1;

# data = data1[:,:-1]
window_size=1000
test_pts=1;500
#************************ starting the window process ***********************************************
# for frame_start in np.arange(0,1000,10):#data.shape[0]-100-window_size):
for frame_start in range(0,1):

    # ipdb.set_trace()
    # L = np.max([data1.shape,data2.shape,data3.shape,data4.shape,data5.shape])


    data_curr = data[frame_start:frame_start+window_size,:]
    # mask = np.zeros(window_size)
    mask_i = np.arange(0,window_size,1)

    sampling = random.choices(mask_i, k=test_pts)
    mask = np.zeros(window_size)>5
    mask[sampling] = True
    # Xpredict = data_curr[mask,0]
    # Ypredict = data_curr[mask,1]
    # XX = data_curr[np.invert(mask),0]
    # YY = data_curr[np.invert(mask),1]
    # XX=XX.reshape(-1,1);
    # YY=YY.reshape(-1,1)
    # data_test = data_curr[]
    # ipdb.set_trace()

    # from sklearn.gaussian_process import GaussianProcessRegressor
    XX = data[:,0].reshape(-1,1);
    XX = data_curr[:,0].reshape(-1,1);
    XX=XX*1.0;
    #
    #
    YY = data_curr[:,1].reshape(-1,1);
    # Xpredict = data[frame_start+window_size:frame_start+window_size+15,0];
    # Ypredict = data[frame_start+window_size:frame_start+window_size+15,1];
    sigma_f = -1;-2.09;-2;
    sigma_l=-3;-7.6;-4;
    sigma_n = -1;np.Inf;-1;-6.28#-np.Inf;#1;#-2;
    eta = 1e-1;
    f=1;
    fprev=0.5;

    L = XX.size
    K=np.zeros([L,L]);
    k = np.zeros([L,L]);
    kl = np.zeros([L,L]);
    tol=1e-3;
    dPdf = np.array([[1]]); dPdl = np.array([[1]]); dPdn = np.array([[1e4]]);
    err = 1;#(np.abs(dPdf[0,0])+np.abs(dPdl[0,0])+np.abs(dPdn[0,0]));
    err_prev = 10;
    err_prev_2 = 100;
    err_grad = tol*2;
    count=0;

    Xii = np.multiply(XX,XX);
    Xii = np.matlib.repmat(Xii,1,L);
    Xjj = Xii.transpose();
    # ipdb.set_trace();
    XXi_XXj = Xii+Xjj-2*XX.dot(XX.transpose());


    res = minimize(nll_fn(XX, YY), [1, 1, -3],method='L-BFGS-B')
               # bounds=((1e-5, None , 0), (1e-5, None, 0)),
               # method='L-BFGS-B')
ipdb.set_trace();
