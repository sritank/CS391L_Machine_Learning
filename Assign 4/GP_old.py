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

marker = '0';
marker_x = marker+'_x';
marker_y = marker+'_y';
marker_z = marker+'_z';
marker_c = marker+'_c';
x_t = 'frame'

#***************Reading data from file********************************************
data = np.array([0 , 0]);
counter = 0
frame_size=100
with open('./data_GP/AG/block1-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213203046-59968-right-speed_0.500.csv', newline='') as csvfile:
     reader = csv.DictReader(csvfile)
     for row in reader:
         counter =counter+1;
         # print(row['frame'], row['0_x'])
         # ipdb.set_trace();
         if(np.float64(row[marker_c])>0 and counter<frame_size):
             data = np.vstack([data, np.array([row[x_t], row[marker_x]],dtype=float)]);

data=np.delete(data,0,0);

# counter=0
# with open('./data_GP/AG/block2-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204004-59968-right-speed_0.500.csv', newline='') as csvfile:
#      reader = csv.DictReader(csvfile)
#      for row in reader:
#          counter =counter+1;
#          # print(row['frame'], row['0_x'])
#          if(np.float64(row[marker_c])>0 and counter<frame_size):
#             data = np.vstack([data, np.array([row[x_t], row[marker_x]],dtype=float)]);




# with open('./data_GP/AG/block3-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204208-59968-right-speed_0.500.csv', newline='') as csvfile:
#      reader = csv.DictReader(csvfile)
#      for row in reader:
#          # print(row['frame'], row['0_x'])
#          if(np.float64(row[marker_c])>0):
#             data = np.vstack([data, np.array([row[x_t], row[marker_x]],dtype=float)]);
#
#
#
# with open('./data_GP/AG/block4-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204925-59968-right-speed_0.500.csv', newline='') as csvfile:
#      reader = csv.DictReader(csvfile)
#      for row in reader:
#          # print(row['frame'], row['0_x'])
#          if(np.float64(row[marker_c])>0):
#             data = np.vstack([data, np.array([row[x_t], row[marker_x]],dtype=float)]);
#
#
# with open('./data_GP/AG/block5-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213210121-59968-right-speed_0.500.csv', newline='') as csvfile:
#      reader = csv.DictReader(csvfile)
#      for row in reader:
#          # print(row['frame'], row['0_x'])
#          if(np.float64(row[marker_c])>0):
#             data = np.vstack([data, np.array([row[x_t], row[marker_x]],dtype=float)]);


from sklearn.gaussian_process import GaussianProcessRegressor
# XX = data[:,0].reshape(-1,1);
XX = data[:,0].reshape(-1,1);
XX=XX*1.0;

# XX = np.array([-2.1, -1.5, -0.7, 0.3, 1.0, 1.8, 2.5]).reshape(-1,1);

YY = data[:,1].reshape(-1,1);
# YY=np.array([-1.5128756 , 0.52371713, -0.1382640378102619, -0.13952425, 0.4967141530112327, -0.93665367, -1.29343995]).reshape(-1,1);
# YY=np.multiply(XX,XX/100);
# YY=XX;
gpr = GaussianProcessRegressor().fit(XX, YY);

l_opt = gpr.kernel_.k2.get_params()['length_scale']
sigma_f_opt = np.sqrt(gpr.kernel_.k1.get_params()['constant_value'])

# X = data[:,0];
# X = np.vstack([X*0+1, X]).transpose();

sigma_f = -1.5;
sigma_l=-5;
sigma_n = -5#-np.Inf;#1;#-2;
eta = 1e-2;

L = XX.size
# ipdb.set_trace();
K=np.zeros([L,L]);
k = np.zeros([L,L]);
kl = np.zeros([L,L]);
tol=1;
dPdf = np.array([[1]]); dPdl = np.array([[1]]); dPdn = np.array([[1]]);
err = (np.abs(dPdf[0,0])+np.abs(dPdl[0,0])+np.abs(dPdn[0,0]));
count=0;

Xii = np.multiply(XX,XX);
Xii = np.matlib.repmat(Xii,1,L);
Xjj = Xii.transpose();
# ipdb.set_trace();
XXi_XXj = Xii+Xjj-2*XX.dot(XX.transpose());


# for i in range(0,L):
#     for j in range(0,L):
#         XXi_XXj[i,j] = (XX[i]-XX[j])**2;



while err >tol:
    if count>1e4:
        break
    # for i in range(0,L):
    #     for j in range(0,L):
    #         k[i,j] = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*(XX[i]-XX[j])*(XX[i]-XX[j]))
    #         kl[i,j] = -0.5*np.exp(sigma_l)*(XX[i]-XX[j])*(XX[i]-XX[j])

    # Xii = np.multiply(XX,XX);
    # Xii = np.matlib.repmat(Xii,1,L);
    # Xjj = Xii.transpose();
    # # ipdb.set_trace();
    # XXi_XXj = Xii+Xjj-2*XX.dot(XX.transpose());
    # sqdist = np.sum(XX**2, 1).reshape(-1, 1) + np.sum(XX**2, 1) - 2 * np.dot(XX, XX.T)

    # l=np.exp(sigma_l);
    # sigma = np.exp(2*sigma_f);
    # sig_eps = np.exp(sigma_n);

    kp = np.exp(sigma_f)*np.exp(-0.5*np.exp(sigma_l)*XXi_XXj);
    # kp1 = np.exp(sigma_f) * np.exp( -0.5 * np.exp(sigma_l) * np.subtract.outer(XX, XX)**2)
    # ipdb.set_trace();
    # kp = sigma*np.exp(-0.5/l*XXi_XXj);
    kl = -0.5*np.exp(sigma_l)*XXi_XXj;
    # kl = 1/(l*l)*XXi_XXj;

    Q = kp + np.eye(L)*np.exp(sigma_n);
    Lc=np.linalg.cholesky(Q)
    # beta=np.linalg.solve(Lc.transpose(),np.linalg.solve(Lc,YY))
    Qinv = np.linalg.solve(Lc.transpose(),np.linalg.solve(Lc,np.eye(L)))
    # Qinv = LA.inv(Q);

    dPdf = 0.5*(YY.transpose()).dot(Qinv).dot(kp).dot(Qinv).dot(YY) - 0.5*np.trace(Qinv.dot(kp));
    # dPdf = 0.5*np.matmul(S,kp).dot(S.transpose()) - 0.5*np.trace(np.matmul(Qinv,kp));
    # ipdb.set_trace();
    # dPdf = 0.5*(YY.transpose()).dot(Qinv).dot(kp).dot(Qinv).dot(YY) - 0.5*np.trace(Qinv.dot(kp));

    dPdl = 0.5*(YY.transpose()).dot(Qinv).dot(np.multiply(kp,kl)).dot(Qinv).dot(YY) - 0.5*np.trace(Qinv.dot(np.multiply(kp,kl)));

    dPdn = 0.5*(YY.transpose()).dot(Qinv).dot(Qinv).dot(YY)*np.exp(sigma_n) - 0.5*np.trace(Qinv*np.exp(sigma_n));

    sigma_f = sigma_f + eta**2*dPdf;#-eta*dPdf doesn't converge
    sigma_l = sigma_l + eta**2*dPdl;#-eta*dPdl doesn't converge
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




# while err >tol:
#
#     if count>3000:
#         break
#
#     kp = sigma_f**2*np.exp(-(0.5/sigma_l**2)*XXi_XXj);
#
#     ipdb.set_trace();
#     # kp = sigma*np.exp(-0.5/l*XXi_XXj);
#     kl = (1/sigma_l**3)*XXi_XXj;
#     # kl = 1/(l*l)*XXi_XXj;
#
#     Q = kp + np.eye(L)*(sigma_n**2);
#     Lc=np.linalg.cholesky(Q)
#     # beta=np.linalg.solve(Lc.transpose(),np.linalg.solve(Lc,YY))
#     Qinv = np.linalg.solve(Lc.transpose(),np.linalg.solve(Lc,np.eye(L)))
#     # Qinv = LA.inv(Q);
#     S = np.matmul(YY.transpose(), Qinv);
#
#     yTQinv = np.matmul(YY.transpose(),Qinv);
#     QinvyT = np.matmul(Qinv,YY);
#     dQdf = 2*sigma_f*np.exp(-(0.5/sigma_l**2)*XXi_XXj);
#     dQdl = np.multiply(kp,kl)
#     dQdn = 2*sigma_n*eye(L)
#     # dPdf = 0.5*(YY.transpose()).dot(Qinv).dot().dot(Qinv).dot(YY) - 0.5*np.trace(Qinv.dot(2*sigma_f*np.exp(-(0.5/sigma_l**2)*XXi_XXj)));
#     dPdf = 0.5*np.matmul(np.matmul(yTQinv,dQdf),QinvyT) - 0.5*np.trace(np.matmul(Qinv,dQdf))
#     # dPdf = 0.5*np.matmul(S,kp).dot(S.transpose()) - 0.5*np.trace(np.matmul(Qinv,kp));
#     # ipdb.set_trace();
#     # dPdf = 0.5*(YY.transpose()).dot(Qinv).dot(kp).dot(Qinv).dot(YY) - 0.5*np.trace(Qinv.dot(kp));
#
#     # dPdl = 0.5*(YY.transpose()).dot(Qinv).dot(np.multiply(kp,kl)).dot(Qinv).dot(YY) - 0.5*np.trace(Qinv.dot(np.multiply(kp,kl)));
#     dPdl = 0.5*np.matmul(np.matmul(yTQinv,dQdl),QinvyT) - 0.5*np.trace(np.matmul(Qinv,dQdl))
#
#
#     # dPdn = 0.5*(YY.transpose()).dot(Qinv).dot(Qinv).dot(YY)*2*sigma_n - 0.5*np.trace(Qinv*2*sigma_n);
#     dPdn = 0.5*np.matmul(np.matmul(yTQinv,dQdn),QinvyT) - 0.5*np.trace(np.matmul(Qinv,dQdn))
#
#
#
#     sigma_f = sigma_f - eta*dPdf;
#     sigma_l = sigma_l - eta*dPdl;
#     # sigma_n = sigma_n - eta*dPdn;
#     err = (np.abs(dPdf[0,0])+np.abs(dPdl[0,0])+np.abs(dPdn[0,0]));
#     count=count+1;
#     print("count=",count)
#     print("err=",err)
#     print("sigma_l=",sigma_l)
#     print("sigma_n=",sigma_n)
#     print("sigma_f=",sigma_f)
#     print("det=",LA.det(Q))
#     # ipdb.set_trace();
#     count=count+1

# ipdb.set_trace();







#*********** Predicting new values ***************************

# sigma_f = 0;#1;
# sigma_l=np.log(10);#2;
# sigma_n = -np.Inf;#1;#-2;

Xstar = np.linspace(XX[0],XX[-1],1000);#XX+0.5;#np.array([1.5, 2.5, 3.5, 4.5, 5.5, 9.5, 15.5, 17.5, 18.5])
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
# KXXstar =

Ym = gpr.predict(Xstar.T,return_std=True);

c1=np.diagonal(PXstar)#.reshape(-1,1);
# plt.plot(data[:,0],data[:,1],marker='o', markerfacecolor='blue', markersize=10);
# plt.scatter(data[:,0],data[:,1],s=7, c='blue', marker='o');

# plt.plot(Xstar,mXstar.T)
# plt.errorbar(Xstar,mXstar, yerr=np.diagonal(PXstar),capsize=0)
# plt.errorbar(Xstar,mXstar, yerr=c1)
plt.errorbar(Xstar.T,mXstar, yerr=np.diagonal(PXstar))
# plt.errorbar(Xstar.T,mXstar, yerr=np.vstack([c1.T, -c1.T]))

# plt.scatter(Xstar,mXstar.T,s=7, c='blue', marker='*')
plt.plot(Xstar.T,Ym[0],'ro');
plt.plot(XX,YY,'g--')
plt.savefig('./plots/marker_0_x.png');
plt.show()
ipdb.set_trace();
plt.close();






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
