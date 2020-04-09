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

marker = '15';
marker_x = marker+'_x';
marker_y = marker+'_y';
marker_z = marker+'_z';
marker_c = marker+'_c';
x_t = 'frame'

#***************Reading data from file********************************************


window_size=30
sigma_l_arr = 0;
sigma_f_arr = 0;
sigma_n_arr = 0;

for frame_start in range(0,20):
    counter = 0;
    data1 = np.array([0 , 0]);
    data2 = np.array([0 , 0]);
    data3 = np.array([0 , 0]);
    data4 = np.array([0 , 0]);
    data5 = np.array([0 , 0]);
    with open('./data_GP/AG/block1-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213203046-59968-right-speed_0.500.csv', newline='') as csvfile:
         reader = csv.DictReader(csvfile)
         for row in reader:
             counter =counter+1;
             # print(row['frame'], row['0_x'])
             # ipdb.set_trace();
             if(np.float64(row[marker_c])>0 and counter>=frame_start and counter<frame_start+window_size):
                 data1 = np.vstack([data1, np.array([row[x_t], row[marker_x], row[marker_c]],dtype=float)]);

    data1=np.delete(data1,0,0);

    counter=0
    with open('./data_GP/AG/block2-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204004-59968-right-speed_0.500.csv', newline='') as csvfile:
         reader = csv.DictReader(csvfile)
         for row in reader:
             counter =counter+1;
             # print(row['frame'], row['0_x'])
             if(np.float64(row[marker_c])>0 and counter>=frame_start and counter<frame_start+window_size):
                data2 = np.vstack([data2, np.array([row[x_t], row[marker_x], row[marker_c]],dtype=float)]);

    data2=np.delete(data2,0,0);

    counter=0
    with open('./data_GP/AG/block3-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204208-59968-right-speed_0.500.csv', newline='') as csvfile:
         reader = csv.DictReader(csvfile)
         for row in reader:
             counter =counter+1;
             # print(row['frame'], row['0_x'])
             if(np.float64(row[marker_c])>0 and counter>=frame_start and counter<frame_start+window_size):
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
             if(np.float64(row[marker_c])>0 and counter>=frame_start and counter<frame_start+window_size):
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
             if(np.float64(row[marker_c])>0 and counter>=frame_start and counter<frame_start+window_size):
                data5 = np.vstack([data5, np.array([row[x_t], row[marker_x], row[marker_c]],dtype=float)]);
    data5=np.delete(data5,0,0);
    ipdb.set_trace()
    L = np.max([data1.shape,data2.shape,data3.shape,data4.shape,data5.shape])
    data=data1;

    from sklearn.gaussian_process import GaussianProcessRegressor
    # XX = data[:,0].reshape(-1,1);
    XX = data[:,0].reshape(-1,1);
    XX=XX*1.0;


    YY = data[:,1].reshape(-1,1);

    sigma_f = 2;
    sigma_l=0;
    sigma_n = -3#-np.Inf;#1;#-2;
    eta = 1e-3;

    L = XX.size
    K=np.zeros([L,L]);
    k = np.zeros([L,L]);
    kl = np.zeros([L,L]);
    tol=2;
    dPdf = np.array([[1]]); dPdl = np.array([[1]]); dPdn = np.array([[1]]);
    err = (np.abs(dPdf[0,0])+np.abs(dPdl[0,0])+np.abs(dPdn[0,0]));
    count=0;

    Xii = np.multiply(XX,XX);
    Xii = np.matlib.repmat(Xii,1,L);
    Xjj = Xii.transpose();
    # ipdb.set_trace();
    XXi_XXj = Xii+Xjj-2*XX.dot(XX.transpose());



    while err >tol:
        if count>1e4:
            break

        kp = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*XXi_XXj);

        kl = -0.5*np.exp(sigma_l)*XXi_XXj;

        Q = kp + np.eye(L)*np.exp(sigma_n);
        Lc=np.linalg.cholesky(Q)

        Qinv = np.linalg.solve(Lc.transpose(),np.linalg.solve(Lc,np.eye(L)))

        dPdf = 0.5*(YY.transpose()).dot(Qinv).dot(kp).dot(Qinv).dot(YY) - 0.5*np.trace(Qinv.dot(kp));

        dPdl = 0.5*(YY.transpose()).dot(Qinv).dot(np.multiply(kp,kl)).dot(Qinv).dot(YY) - 0.5*np.trace(Qinv.dot(np.multiply(kp,kl)));

        dPdn = 0.5*(YY.transpose()).dot(Qinv).dot(Qinv).dot(YY)*np.exp(sigma_n) - 0.5*np.trace(Qinv*np.exp(sigma_n));

        sigma_f = sigma_f + eta*dPdf;#-eta*dPdf doesn't converge
        sigma_l = sigma_l + eta*dPdl;#-eta*dPdl doesn't converge
        sigma_n = sigma_n + eta*dPdn;#-eta*dPdn doesn't converge
        err = (np.abs(dPdf[0,0])+np.abs(dPdl[0,0])+np.abs(dPdn[0,0]));
        count=count+1;
        # print("count=",count)
        # print("err=",err)
        # print("sigma_l=",sigma_l)
        # print("sigma_n=",sigma_n)
        # print("sigma_f=",sigma_f)
        # print("det=",LA.det(Q))
        # ipdb.set_trace();

    print("count=",count)
    print("err=",err)
    print("sigma_l=",sigma_l)
    print("sigma_n=",sigma_n)
    print("sigma_f=",sigma_f)
    print("det=",LA.det(Q))

    sigma_n_arr = np.vstack([sigma_n_arr, sigma_n]);
    sigma_f_arr = np.vstack([sigma_f_arr, sigma_f]);
    sigma_l_arr = np.vstack([sigma_l_arr, sigma_l]);




#*********** Predicting new values ***************************

# sigma_f = 0;#1;
# sigma_l=np.log(10);#2;
# sigma_n = -np.Inf;#1;#-2;
sigma_l_arr = np.delete(sigma_l_arr,0)
sigma_f_arr = np.delete(sigma_f_arr,0)
sigma_n_arr = np.delete(sigma_n_arr,0)


Xstar = np.linspace(XX[0],XX[-1],1000);#XX+0.5;
Xstar = Xstar.reshape(-1,1)
Xstar = Xstar.transpose()
L1 = Xstar.size
XXstar1 = np.matlib.repmat(XX,1,L1)
XXstar1 = np.multiply(XXstar1,XXstar1)
XXstar2 = np.matlib.repmat(Xstar,L,1)
XXstar2 = np.multiply(XXstar2,XXstar2)
XXstar = XXstar1+XXstar2-2*XX.dot(Xstar)

Xstar_cov = np.multiply(np.matlib.repmat(Xstar,L1,1),np.matlib.repmat(Xstar,L1,1)) + (np.multiply(np.matlib.repmat(Xstar,L1,1),np.matlib.repmat(Xstar,L1,1))).transpose() -2*Xstar.T.dot(Xstar);

KXstarX = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*XXstar.transpose())
kp = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*XXi_XXj) + np.eye(L)*np.exp(sigma_n);
mXstar = KXstarX.dot(LA.inv(kp)).dot(YY)
PXstar = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*Xstar_cov) - KXstarX.dot(LA.inv(kp)).dot(KXstarX.transpose())


# Ym = gpr.predict(Xstar.T,return_std=True);

c1=np.diagonal(PXstar)#.reshape(-1,1);

plt.errorbar(Xstar.T,mXstar, yerr=np.sqrt(np.diagonal(PXstar))*2)

# plt.plot(Xstar.T,Ym[0],'ro');
plt.plot(XX,YY,'g--')

plt.savefig('./plots/marker_0_x.png');
plt.show()
ipdb.set_trace();
plt.close();
plt.plot(sigma_f_arr,'b')

plt.plot(sigma_l_arr,'g')

plt.plot(sigma_n_arr,'r')
plt.show()

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
