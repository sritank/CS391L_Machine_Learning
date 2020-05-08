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

marker = '8';
marker_x = marker+'_x';
marker_y = marker+'_y';
marker_z = marker+'_z';
marker_c = marker+'_c';
x_t = 'frame'

def exponential_cov(x, y, sigma_f, sigma_l):
    return np.exp(sigma_f) * np.exp( -0.5 * np.exp(sigma_l) * np.subtract.outer(x, y)**2)
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
# data_report = pkl.load(open("./GP_hyperparam_15_x_report.p","rb"));
# data=np.zeros((1000,2))
# data[:,0] = data_report['X'].reshape(-1,)
# data[:,1] = data_report['Y'].reshape(-1,)
# ************** mixing 2 traces **********************************
# counter=0;
# flag=0;
# for i in range(0,1010):
#     r=random.randint(1,2);
#     flag=0;
#     while flag==0:
#         if data1[i,2]<0 and data2[i,2]<0:
#             flag=1;
#             continue;
#         elif r==1 and data1[i,2]>0:
#             data[counter,:] = data1[i,:-1];
#             counter=counter+1;
#             flag=1;
#         elif r==2 and data2[i,2]>0:
#             data[counter,:] = data2[i,:-1];
#             counter=counter+1;
#             flag=1;
#         else:
#             r=random.randint(1,2);


#counter=0;
#for i in range(0,1010):
#    if data1[i,2]<=0:
#        continue;
#    else:
#        data[counter,:] = data1[i,:-1];
#        counter=counter+1;

# data = data1[:,:-1]
window_size=100
test_pts=1;
window_start=0
window_end=1000-window_size
delta=5
logP_local=[0];
#************************ starting the window process ***********************************************
for frame_start in np.arange(window_start,window_end,delta):#data.shape[0]-100-window_size):
# for frame_start in range(0,1):

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
    # XX = data[:,0].reshape(-1,1);
    XX = data_curr[:,0].reshape(-1,1);
    XX=XX*1.0;
    # XX = np.array([-2.1, -1.5, -0.7, 0.3, 1.0, 1.8, 2.5]).reshape(-1,1);
    #
    #
    YY = data_curr[:,1].reshape(-1,1);
    # YY=np.array([-1.5128756 , 0.52371713, -0.1382640378102619, -0.13952425, 0.4967141530112327, -0.93665367, -1.29343995]).reshape(-1,1);
    # Xpredict = data[frame_start+window_size:frame_start+window_size+15,0];
    # Ypredict = data[frame_start+window_size:frame_start+window_size+15,1];
    sigma_f = -1;2;#-2.09;-2;
    sigma_l=-8;-6;#-7.6;-4;
    sigma_n = -8;-9;#np.Inf;-1;-6.28#-np.Inf;#1;#-2;
    eta = 1e-2;
    f=1;
    fprev=f/2;

    L = XX.size
    K=np.zeros([L,L]);
    k = np.zeros([L,L]);
    kl = np.zeros([L,L]);
    tol=2;
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

    # ipdb.set_trace()

    # while (err < err_prev and err_prev < err_prev_2) or err_grad>tol:
    # while err < err_prev or err_grad>tol:

    # while f>=fprev:# and count>3:
    while    err_grad > tol:# or np.abs(dPdn[0,0])>500:
        if count>1e4:
            break

        kp = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*XXi_XXj);

        kl = -0.5*np.exp(sigma_l)*XXi_XXj;

        Q = kp + np.eye(L)*np.exp(sigma_n);
        Lc=np.linalg.cholesky(Q)

        Qinv = np.linalg.solve(Lc.transpose(),np.linalg.solve(Lc,np.eye(L)))
        beta = np.linalg.solve(Lc.transpose(), np.linalg.solve(Lc,YY))
        # negLogP = -0.5*YY.T.dot(Qinv).dot(YY) - np.sum(np.log(Lc)) - L/2*np.log(np.pi);
        # negLogP = -0.5*YY.T.dot(beta) - np.sum(np.log(Lc)) - L/2*np.log(np.pi);
        # negLogP = -0.5*YY.T.dot(Qinv).dot(YY) - 0.5*(np.log(LA.det(Q)+sys.float_info.min)) - L/2*np.log(2*np.pi);

        dPdf = 0.5*(YY.transpose()).dot(Qinv).dot(kp).dot(Qinv).dot(YY) - 0.5*np.trace(Qinv.dot(kp));

        dPdl = 0.5*(YY.transpose()).dot(Qinv).dot(np.multiply(kp,kl)).dot(Qinv).dot(YY) - 0.5*np.trace(Qinv.dot(np.multiply(kp,kl)));

        dPdn = 0.5*(YY.transpose()).dot(Qinv).dot(Qinv).dot(YY)*np.exp(sigma_n) - 0.5*np.trace(Qinv)*np.exp(sigma_n);#should be -trace(Qinv) but that doesn't give right answer

        sigma_f = sigma_f + eta*dPdf;#-eta*dPdf doesn't converge
        sigma_l = sigma_l + eta*dPdl;#-eta*dPdl doesn't converge
        sigma_n = sigma_n + eta*dPdn;#-eta*dPdn doesn't converge
#
# #************ Calculate prediction error **************************
#         # Xpredict = Xpredict.reshape(-1,1)
#         # Xpredict = Xpredict.transpose()
#         # Lpredict = Xpredict.size
#         # XXpred1 = np.matlib.repmat(XX,1,Lpredict)
#         # XXpred1 = np.multiply(XXpred1,XXpred1)
#         # XXpred2 = np.matlib.repmat(Xpredict,L,1)
#         # XXpred2 = np.multiply(XXpred2,XXpred2)
#         # XXpred = XXpred1+XXpred2-2*XX.dot(Xpredict)
#
#         # Xpred_cov = np.multiply(np.matlib.repmat(Xpredict,Lpredict,1),np.matlib.repmat(Xpredict,Lpredict,1)) + (np.multiply(np.matlib.repmat(Xpredict,Lpredict,1),np.matlib.repmat(Xpredict,Lpredict,1))).transpose() -2*Xpredict.T.dot(Xpredict);
#
#         # KXpredX = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*XXpred.transpose())
#         # kp = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*XXi_XXj) + np.eye(L)*np.exp(sigma_n);
#         # mXpred = KXpredX.dot(LA.inv(kp)).dot(YY)
#         #
#         # err_prev_2 = err_prev;
#         # err_prev = err;
#         # err = np.sqrt(np.mean(np.multiply(Ypredict-mXpred, Ypredict-mXpred)));
        # err_grad = (np.abs(dPdf[0,0])+np.abs(dPdl[0,0]));#+np.abs(dPdn[0,0]));
        err_grad = (np.abs(dPdf[0,0])+np.abs(dPdl[0,0])+np.abs(dPdn[0,0]));
#
#         fprev=f;
#         f=negLogP;
#         # err_grad = np.abs(dPdl[0,0]);
#         # err_grad=0;
#         # print("count=",count)
        # print("grad=",err_grad)
        count=count+1;
#         # print("count=",count)
#         # print("err=",err)
#         # print("f=",f)
#         # print("sigma_l=",sigma_l)
#         # print("sigma_n=",sigma_n)
#         # print("sigma_f=",sigma_f)
#         # print("det=",LA.det(Q))
#         # ipdb.set_trace();
#
    print("count=",count)
    print("err=",err)
    print("sigma_l=",sigma_l)
    print("sigma_n=",sigma_n)
    print("sigma_f=",sigma_f)
    print("det=",LA.det(Q))
    # if count<5000:
    sigma_n_arr = np.vstack([sigma_n_arr, sigma_n]);
    sigma_f_arr = np.vstack([sigma_f_arr, sigma_f]);
    sigma_l_arr = np.vstack([sigma_l_arr, sigma_l]);
    # negLogP = -0.5*YY.T.dot(Qinv).dot(YY) - 0.5*(np.log(LA.det(Q)+sys.float_info.min)) - L/2*np.log(2*np.pi);
    # logP_local = np.vstack([logP_local, negLogP]);
    #
    # ipdb.set_trace()
#********************** Plotting local kernels, comment if not plotting ***************************
#     Xstar = np.linspace(XX[0],XX[-1],window_size*10);#XX+0.5;
#     # Xstar = np.linspace(-3,3,1000)
#     # Xstar = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 9.5, 15.5, 17.5, 18.5])
#     Xstar = Xstar.reshape(-1,1)
#     Xstar = Xstar.transpose()
#     L1 = Xstar.size
#     XXstar1 = np.matlib.repmat(XX,1,L1)
#     XXstar1 = np.multiply(XXstar1,XXstar1)
#     XXstar2 = np.matlib.repmat(Xstar,L,1)
#     XXstar2 = np.multiply(XXstar2,XXstar2)
#     XXstar = XXstar1+XXstar2-2*XX.dot(Xstar)
#
#     Xstar_cov = np.multiply(np.matlib.repmat(Xstar,L1,1),np.matlib.repmat(Xstar,L1,1)) + (np.multiply(np.matlib.repmat(Xstar,L1,1),np.matlib.repmat(Xstar,L1,1))).transpose() -2*Xstar.T.dot(Xstar);
#
#     KXstarX = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*XXstar.transpose())
#     kp = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*XXi_XXj) + np.eye(L)*np.exp(sigma_n);
#     Lc_kp=np.linalg.cholesky(kp)
#
#     kpinv = np.linalg.solve(Lc_kp.transpose(),np.linalg.solve(Lc_kp,np.eye(L)))
#     mXstar = KXstarX.dot(kpinv).dot(YY)
#
#     PXstar = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*Xstar_cov) + np.eye(L1)*np.exp(sigma_n) - KXstarX.dot(kpinv).dot(KXstarX.transpose())
#
#     plt.errorbar(Xstar.T,mXstar,color=[0,0,0], yerr=np.sqrt(np.diagonal(PXstar))*2, ecolor = [0.7,0.7,0.7], label='GP mean function')
#     plt.plot(Xstar.T,mXstar,color='k')
#
#
# l1,=plt.plot(data[:,0],data[:,1],'r.',label='input data')
#
# plt.title('Local GP kernels fitted to mixture of 5 traces')
# plt.xlabel('Frame number')
# plt.ylabel('Position (m)')
# plt.legend(handles = [l1])
# plt.rcParams.update({'font.size': 25})
# plt.show()

# logP_report_kernels = {"logP": logP_local}
# pkl.dump(logP_report_kernels,open("GP_logP_local.p","wb"))

#***********************************************************************************************

#
#
#
#
# #*********** Predicting new values ***************************
#
# sigma_f = 2; 0;#1;
# sigma_l= -6; np.log(10);#2;
# sigma_n = -9; -np.Inf;#1;#-2;
sigma_l_arr = np.delete(sigma_l_arr,0)
sigma_f_arr = np.delete(sigma_f_arr,0)
sigma_n_arr = np.delete(sigma_n_arr,0)
logP_local = np.delete(logP_local,0)
# ipdb.set_trace();
Xstar = np.linspace(XX[0],XX[-1],window_size*3);#XX+0.5;
# Xstar = np.linspace(-3,3,1000)
# Xstar = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 9.5, 15.5, 17.5, 18.5])
Xstar = Xstar.reshape(-1,1)
Xstar = Xstar.transpose()
L1 = Xstar.size


# B = exponential_cov(Xstar, XX, sigma_f, sigma_l)
# C = exponential_cov(XX, XX, sigma_f, sigma_l)
# A = exponential_cov(Xstar, Xstar, sigma_f, sigma_l)
# B=B.reshape(L1,L)
# C=C.reshape(L,L)
# A=A.reshape(L1,L1)
# #
# # mu = B.dot(np.linalg.inv(C)).dot(YY) #np.linalg.inv(C).dot(B.T).T.dot(YY)
# # mu = np.linalg.inv(C).dot(B.T).T.dot(YY);
# sigma = A - B.dot(np.linalg.inv(C).dot(B.T))



XXstar1 = np.matlib.repmat(XX,1,L1)
XXstar1 = np.multiply(XXstar1,XXstar1)
XXstar2 = np.matlib.repmat(Xstar,L,1)
XXstar2 = np.multiply(XXstar2,XXstar2)
XXstar = XXstar1+XXstar2-2*XX.dot(Xstar)

Xstar_cov = np.multiply(np.matlib.repmat(Xstar,L1,1),np.matlib.repmat(Xstar,L1,1)) + (np.multiply(np.matlib.repmat(Xstar,L1,1),np.matlib.repmat(Xstar,L1,1))).transpose() -2*Xstar.T.dot(Xstar);

KXstarX = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*XXstar.transpose())
kp = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*XXi_XXj) + np.eye(L)*np.exp(sigma_n);
Lc_kp=np.linalg.cholesky(kp)

kpinv = np.linalg.solve(Lc_kp.transpose(),np.linalg.solve(Lc_kp,np.eye(L)))

mXstar = KXstarX.dot(kpinv).dot(YY)

# mXstar = KXstarX.dot(LA.inv(kp)).dot(YY)

PXstar = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*Xstar_cov) + np.eye(L1)*np.exp(sigma_n) - KXstarX.dot(kpinv).dot(KXstarX.transpose())

# PXstar = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*Xstar_cov) - KXstarX.dot(kpinv).dot(KXstarX.transpose())

# Ym = gpr.predict(Xstar.T,return_std=True);

c1=np.diagonal(PXstar)#.reshape(-1,1);

l1,l2,l3 = plt.errorbar(Xstar.T,mXstar,color=[0,0,0], yerr=np.sqrt(np.diagonal(PXstar))*2, ecolor = [0.7,0.7,0.7], label='GP mean function')
# plt.errorbar((Xstar).T,mu, yerr=np.sqrt(np.diagonal(sigma)),color='b')
# plt.plot(Xstar.T,Ym[0],'ro');
# plt.plot(XX,YY,'g--')
# plt.plot(Xstar.T,mu,color='g')
l4,=plt.plot(Xstar.T,mXstar,color='k', label='Optimal GP mean function')
l5, = plt.plot(XX,YY,'r.',label='input data')
plt.title('Optimal GP kernel fitted to mixture of 5 traces')
plt.xlabel('Frame number')
plt.ylabel('Position (m)')
plt.legend(handles = [l4,l5])
plt.rcParams.update({'font.size': 25})
plt.savefig('./plots/marker_0_x.png');
plt.show()
ipdb.set_trace();
plt.close();
l1, = plt.plot(np.arange(window_start,window_end,delta),sigma_f_arr,'b', label='sigma_f')

l2, = plt.plot(np.arange(window_start,window_end,delta),sigma_l_arr,'g', label='sigma_l')

l3, = plt.plot(np.arange(window_start,window_end,delta),sigma_n_arr,'r', label='sigma_n')

l4, = plt.plot(data[:,0],data[:,1],'k', label='trajectory')

# l5, = plt.plot(np.arange(window_start,window_end,delta),sigma_f_arr*0-1.79495408,'--b', label='sigma_f_global')
#
# l6, = plt.plot(np.arange(window_start,window_end,delta),sigma_l_arr*0-8.27030516,'--g', label='sigma_l_global')
#
# l7, = plt.plot(np.arange(window_start,window_end,delta),sigma_n_arr*0-6.33156224,'--r', label='sigma_n_global')

# plt.legend(handles = [l1,l2,l3,l4,l5,l6,l7])
plt.legend(handles = [l1,l2,l3,l4])

plt.xlabel('Frame number')
plt.ylabel('hyperparam value')
plt.xlim(-10,1400)
plt.title('Hyperparams Subject:AG, marker:15_x, Single trace')
plt.rcParams.update({'font.size': 17})
plt.savefig('./plots/marker_'+marker+'_x_hyperparams.png');

plt.show()

plt.hist(sigma_f_arr,bins=70)
plt.xlabel('sigma_f_value')
plt.ylabel('frequency')
plt.title('Histogram of sigma_f')
plt.rcParams.update({'font.size': 17})
plt.show()


plt.hist(sigma_l_arr,bins=70)
plt.xlabel('sigma_l_value')
plt.ylabel('frequency')
plt.title('Histogram of sigma_l')
plt.rcParams.update({'font.size': 17})
plt.show()

plt.hist(sigma_n_arr,bins=70)
plt.xlabel('sigma_n_value')
plt.ylabel('frequency')
plt.title('Histogram of sigma_n')
plt.rcParams.update({'font.size': 17})
plt.show()


l1, = plt.plot(data1[data1[:,2]>0,0],data1[data1[:,2]>0,1], label='trial 1',linewidth='2',color='b')
l2, = plt.plot(data2[data2[:,2]>0,0],data2[data2[:,2]>0,1], label='trial 2',linewidth='2',color='g')
l3, = plt.plot(data3[data3[:,2]>0,0],data3[data3[:,2]>0,1], label='trial 3',linewidth='2',color='r')
l4, = plt.plot(data4[data4[:,2]>0,0],data4[data4[:,2]>0,1], label='trial 4',linewidth='2',color='k')
l5, = plt.plot(data5[data5[:,2]>0,0],data5[data5[:,2]>0,1], label='trial 5',linewidth='2',color='c')
plt.legend(handles = [l1,l2,l3,l4,l5])
plt.title('Subject:AG, marker:15_x, trajectory')
plt.xlabel('Frame number')
plt.ylabel('Position (m)')
plt.rcParams.update({'font.size': 22})
plt.show()

# hyper_param_arr = {"f": sigma_f_arr, "l":sigma_l_arr, "n":sigma_n_arr, "X":data[:,0], "Y":data[:,1]}
hyper_param_arr = {"f": sigma_f_arr, "l":sigma_l_arr, "n":sigma_n_arr, "X":XX, "Y":YY}

pkl.dump(hyper_param_arr,open("GP_hyperparam_0_x.p","wb"));
ipdb.set_trace();








# with open('./data_GP/AG/block1-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213203046-59968-right-speed_0.500.csv') as csv_file:
#
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(row[0], row[11])
#             line_count += 1
#         else:
#             # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
#             data = np.vstack([data, np.array([row[0], row[11]],dtype=float)]);
#             line_count += 1
    # csv_dict = csv.DictReader(csv_file);
