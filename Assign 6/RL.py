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
from sklearn.preprocessing import normalize

H=7; L=25;
W=9; D=25;

#****************module for staying on sidewalk*****************
G = np.zeros((H,L));
Q = np.random.rand(H*L,H*L)*1e-3;
T = np.zeros((H*L,H*L));
H=6; L=3;
sidewalk_penalty = 50.;
sidewalk_reward = 5.;
R = np.zeros((H,L))-sidewalk_penalty;
Q = np.random.rand(H,L)*1e-3;
#R = np.zeros((16,3))
#R states are 000, 001, 011, 111, 110, 100; 1=on sidewalk, 0=not on sidewalk
#actions are move NE, move E, move SE
dict_s = {'000':0, '001':1, '011':2, '111':3, '110':4, '100':5}

R[1,2] = sidewalk_reward;
R[2,1:] = sidewalk_reward;
R[3,:] = sidewalk_reward;
R[4,:2] = sidewalk_reward;
R[5,0] = sidewalk_reward;
epsilon = 0.2;
alpha = 0.3;
gamma = 0.8;

# ipdb.set_trace()


deltaQ_norm = 1;
Q_prev = Q*1.0;
st = random.randint(0,H-1);
iter=0;
iter_min=500;
iter_max=1000;
tol=1e-3;
O = np.random.randint(2, size=(W, D))*0
O[3:6,:]=1;
# st_ac_pair_picked=0;
while iter<iter_max:
    start = random.randint(2,6)
    jj=0;
    i=start;

    if deltaQ_norm<tol and iter>iter_min:
        break
    while  jj<D-1:
        st_int = str(O[i+1,jj+1])+ str(O[i,jj+1]) + str(O[i-1,jj+1]);
        st = dict_s[st_int];
        val_act_picked=0;
        # ipdb.set_trace()
        while val_act_picked!=1:
            if np.random.rand()<epsilon:
                ac = random.randint(0,L-1);
            else:
                ac = np.argmax(Q[st,:]);
            if (i==W-2 and ac==0) or (i==1 and ac==2):
                val_act_picked=0;
            else:
                val_act_picked=1;

        if ac==0:
            i=i+1;
        elif ac==2:
            i=i-1;
        jj=jj+1;
        # if st==5 and ac==2:
        #     st_ac_pair_picked=1;
        # sprime = random.randint(0,H-1);
        if jj==D-1:
            next_st_int = '111';
        else:
            next_st_int = str(O[i+1,jj+1])+ str(O[i,jj+1]) + str(O[i-1,jj+1]);
        sprime = dict_s[next_st_int];
        Q[st,ac] = (1-alpha)*Q[st,ac] + alpha*(R[st,ac] + gamma*max(Q[sprime,:]));
    iter=iter+1;
    deltaQ = np.abs(Q-Q_prev);
    deltaQ_norm = np.mean(deltaQ);
    # print("dQ = ", deltaQ_norm)
    Q_prev = Q*1.0;

Q_swalk = Q;

# ipdb.set_trace()
#****************module for avoiding obstacles*****************
W=7; D=25;
H=8; L=3;
obstacle_penalty = -100;
obstacle_reward = 0.;
R = np.zeros((H,L)) + obstacle_reward;
Q = np.random.rand(H,L)*1e-3;
#R = np.zeros((16,3))
#R states are 000, 001, 010, 011, 111, 110, 101, 100; 1=obstacle, 0=no obstacle
#R actions are move NE, move E, move SE
R[0,1] = 10.;
R[1,2] = obstacle_penalty;
R[2,1] = obstacle_penalty;
R[3,1:] = obstacle_penalty;
R[4,:] = obstacle_penalty;
R[5,:2] = obstacle_penalty;
R[6,[0,2]] = obstacle_penalty;
R[7,0] = obstacle_penalty;
epsilon = 0.1;
alpha = 0.2;
gamma = 0.8;

# ipdb.set_trace()


deltaQ_norm = 1;
Q_prev = Q*1.0;
st = random.randint(0,H-1);
iter=0;
tol=1e-3;
iter_min=100;
iter_max = 1000;
# while deltaQ_norm>tol:
dict_o = {'000':0, '001':1, '010':2, '011':3, '111':4, '110':5, '101':6, '100':7}


# while iter<iter_max:
#     if deltaQ_norm<tol and iter>iter_min:
#         break
#     # ipdb.set_trace()
#     if np.random.rand()<epsilon:
#         ac = random.randint(0,L-1);
#     else:
#         ac = np.argmax(Q[st,:]);
#     if ac==0:
#         i=i+1;
#     elif ac==2:
#         i=i-1;
#     j=j+1;
#     st_int = np.array([O[i,j+1]])
#     sprime = random.randint(0,H-1);
#     Q[st,ac] = (1-alpha)*Q[st,ac] + alpha*(R[st,ac] + gamma*max(Q[sprime,:]));
#     st = sprime;
#     iter=iter+1;
#     deltaQ = np.abs(Q-Q_prev);
#     deltaQ_norm = np.mean(deltaQ);
#     print("dQ = ", deltaQ_norm)
#     Q_prev = Q*1.0;

while iter<iter_max:
    start = random.randint(2,4)
    jj=0;
    i=start;
    O = np.random.randint(6, size=(W, D))
    O[O!=1]=0;
    if deltaQ_norm<tol and iter>iter_min:
        break
    while jj<D-1:
        st_int = str(O[i+1,jj+1])+ str(O[i,jj+1]) + str(O[i-1,jj+1]);
        st = dict_o[st_int];
        val_act_picked=0;
        # if deltaQ_norm<tol and iter>iter_min:
        #     break
        # ipdb.set_trace()
        while val_act_picked!=1:
            if np.random.rand()<epsilon:
                ac = random.randint(0,L-1);
            else:
                ac = np.argmax(Q[st,:]);
            if (i==W-2 and ac==0) or (i==1 and ac==2):
                val_act_picked=0;
            else:
                val_act_picked=1;

        if ac==0:
            i=i+1;
        elif ac==2:
            i=i-1;
        jj=jj+1;
        # sprime = random.randint(0,H-1);
        if jj==D-1:
            next_st_int = '000';
        else:
            next_st_int = str(O[i+1,jj+1])+ str(O[i,jj+1]) + str(O[i-1,jj+1]);
        sprime = dict_o[next_st_int];
        Q[st,ac] = (1-alpha)*Q[st,ac] + alpha*(R[st,ac] + gamma*max(Q[sprime,:]));
        # st = sprime;
    iter=iter+1;
    deltaQ = np.abs(Q-Q_prev);
    deltaQ_norm = np.mean(deltaQ);
    # print("dQ = ", deltaQ_norm)
    Q_prev = Q*1.0;

Q_obstacle = Q;

# ipdb.set_trace()

#***************Litter module****************************

H=8; L=3;
# litter_penalty = -100;
litter_reward = 100.;
R = np.zeros((H,L));
Q = np.random.rand(H,L)*1e-3;
#R = np.zeros((16,3))
#R states are 000, 001, 010, 011, 111, 110, 101, 100; 1=litter, 0=no litter
#R actions are move NE, move E, move SE
R[0,1] = 40.;#reward for just moving forward
R[1,2] = litter_reward;
R[2,1] = litter_reward;
R[3,1:] = litter_reward;
R[4,:] = litter_reward;
R[5,:2] = litter_reward;
R[6,[0,2]] = litter_reward;
R[7,0] = litter_reward;
epsilon = 0.1;
alpha = 0.4;
gamma = 0.5;

# ipdb.set_trace()


deltaQ_norm = 1;
Q_prev = Q*1.0;
st = random.randint(0,H-1);
iter=0;
tol=1e-3;
iter_min=100;
iter_max = 1000;
# while deltaQ_norm>tol:
dict_l = {'000':0, '001':1, '010':2, '011':3, '111':4, '110':5, '101':6, '100':7}

while iter<iter_max:
    start = random.randint(2,4)
    jj=0;
    i=start;
    O = np.random.randint(6, size=(W, D))
    O[O!=1]=0;
    if deltaQ_norm<tol and iter>iter_min:
        break
    while jj<D-1:
        st_int = str(O[i+1,jj+1])+ str(O[i,jj+1]) + str(O[i-1,jj+1]);
        # st_int = str(O[i+1,jj])+ str(O[i,jj+1]) + str(O[i-1,jj]);
        st = dict_l[st_int];
        val_act_picked=0;
        if deltaQ_norm<tol and iter>iter_min:
            break
        # ipdb.set_trace()
        while val_act_picked!=1:
            if np.random.rand()<epsilon:
                ac = random.randint(0,L-1);
            else:
                ac = np.argmax(Q[st,:]);
            if (i==W-2 and ac==0) or (i==1 and ac==2):
                val_act_picked=0;
            else:
                val_act_picked=1;

        if ac==0:
            i=i+1;
        elif ac==2:
            i=i-1;
        # else:
            # jj=jj+1;
        jj=jj+1;
        if jj==D-1:
            next_st_int = '000';
            # next_st_int = str(O[i+1,jj])+ '0' + str(O[i-1,jj]);
        else:
            next_st_int = str(O[i+1,jj+1])+ str(O[i,jj+1]) + str(O[i-1,jj+1]);
            # next_st_int = str(O[i+1,jj])+ str(O[i,jj+1]) + str(O[i-1,jj]);
        sprime = dict_l[next_st_int];
        Q[st,ac] = (1-alpha)*Q[st,ac] + alpha*(R[st,ac] + gamma*max(Q[sprime,:]));
        if O[i,jj]==1:
            O[i,jj]=0; #litter has been picked up
        # st = sprime;
    iter=iter+1;
    deltaQ = np.abs(Q-Q_prev);
    deltaQ_norm = np.mean(deltaQ);
    # print("dQ = ", deltaQ_norm)
    Q_prev = Q*1.0;

Q_litter = Q;

# ipdb.set_trace()

#***********************Generating map and executing agent***************************
H=7; L=25;
W=7; D=25;
M = np.zeros((H,L));


M = np.random.randint(8, size=(W, D))
M[M>2]=0;

M_o = M*1;
M_o[M_o==1]=0;
M_o[M_o==2]=1;
M_o[:,0]=0;#so that agent does not begin from an obstacle


M_l = M*1;
M_l[M_l==2]=0;

M_s = M*0;
M_s[2:5,:]=1;

#***********************Normalizing Q matrices***********************
# Q_swalk_rowsums = Q_swalk.sum(axis=1);
# Q_swalk = Q_swalk/ Q_swalk_rowsums[:,np.newaxis];
Q_swalk = normalize(Q_swalk, axis=1, norm='l1');
Q_obstacle = normalize(Q_obstacle, axis=1, norm='l1');
Q_litter = normalize(Q_litter, axis=1, norm='l1');

start = random.randint(2,W-3);
jj=0;
i=start;
agent_i = i;
agent_j = jj;
Q_ac = np.zeros(3);
# ipdb.set_trace()
w_s = 5; w_o=5; w_l=8;
while jj<D-1:
    st_ind_o = str(M_o[i+1,jj+1])+ str(M_o[i,jj+1]) + str(M_o[i-1,jj+1]);
    st_o = dict_o[st_ind_o];

    st_ind_l = str(M_l[i+1,jj+1])+ str(M_l[i,jj+1]) + str(M_l[i-1,jj+1]);
    st_l = dict_l[st_ind_l];

    st_ind_s = str(M_s[i+1,jj+1])+ str(M_s[i,jj+1]) + str(M_s[i-1,jj+1]);
    st_s = dict_s[st_ind_s];

    Q_ac[0] = w_s*Q_swalk[st_s,0] + w_o*Q_obstacle[st_o,0] + w_l*Q_litter[st_l,0];
    Q_ac[1] = w_s*Q_swalk[st_s,1] + w_o*Q_obstacle[st_o,1] + w_l*Q_litter[st_l,1];
    Q_ac[2] = w_s*Q_swalk[st_s,2] + w_o*Q_obstacle[st_o,2] + w_l*Q_litter[st_l,2];
    print(i)
    ac = np.argmax(Q_ac);
    if ac==0 and i>=W-2:
        ac=np.argmax(Q_ac[1:])+1
    if ac==2 and i<=1:
        ac=np.argmax(Q_ac[:2])

    if ac==0:
        i=i+1;
    elif ac==2:
        i=i-1;
    jj=jj+1;


    agent_i = np.vstack([agent_i,i]);
    agent_j = np.vstack([agent_j,jj]);




plt.spy(np.ones((W,D)), marker='o', markersize=6, color='k')
plt.spy(M_s, marker='o', markersize=6, color='g')
plt.spy(M_o, marker='x', markersize=10, color='r')
plt.spy(M_l, marker='s', markersize=10, color='b')
# plt.plot(agent_j, W-agent_i-1, '--k')
plt.plot(agent_j, agent_i, '--k')
plt.show()

ipdb.set_trace()
# for i in range(0,H):
#     for j in range(0,L):
#         # T(i,j)=1;
#         if i==0:
#             T[i,L+j]=1;
#         if j==0:
#             T[i,j+1]=1;
#         if i==H-1:
#             T[i,(H-2)*L+j]=1;
#         if j==L-1:
#             T[i,i*L+j-1]=1;
#         if i<H-1 and i>0 and j<L-1 and j>0:
#             T[i,i*L+j+1]=1;
#             T[i,(i-1)*L+j]=1;
#             T[i,(i+1)*L+j]=1;
# Q[0:1,:] = -50;
# Q[4:5,:] = -50;
# Q[2:3,:] = 2;
