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

H=7; L=25;
W=7; D=25;

#****************module for staying on sidewalk*****************
G = np.zeros((H,L));
Q = np.random.rand(H*L,H*L)*1e-3;
T = np.zeros((H*L,H*L));
H=6; L=3;
sidewalk_penalty = 5;
sidewalk_reward = 5;
R = np.zeros((H,L))-sidewalk_penalty;
Q = np.random.rand(H,L)*1e-3;
#R = np.zeros((16,3))
#R states are 000, 001, 011, 111, 110, 100; 1=on sidewalk, 0=not on sidewalk
#R actions are move NE, move E, move SE

R[1,2] = sidewalk_reward;
R[2,1:] = sidewalk_reward;
R[3,:] = sidewalk_reward;
R[4,:2] = sidewalk_reward;
R[5,0] = sidewalk_reward;
epsilon = 0.2;
alpha = 0.1;
gamma = 0.4;




deltaQ_norm = 1;
Q_prev = Q*1.0;
st = random.randint(0,H-1);
iter=0;
tol=1e-3;
while deltaQ_norm>tol:
# while iter<1000:
    if iter>1000:
        break
    # ipdb.set_trace()
    if np.random.rand()<epsilon:
        ac = random.randint(0,L-1);
    else:
        ac = np.argmax(Q[st,:]);
    sprime = random.randint(0,H-1);
    Q[st,ac] = (1-alpha)*Q[st,ac] + alpha*(R[st,ac] + gamma*max(Q[sprime,:]));
    st = sprime;
    iter=iter+1;
    deltaQ = np.abs(Q-Q_prev);
    deltaQ_norm = np.mean(deltaQ);
    print("dQ = ", deltaQ_norm)
    Q_prev = Q*1.0;

Q_swalk = Q;

#****************module for avoiding obstacles*****************

H=8; L=3;
obstacle_penalty = -10;
obstacle_reward = 0.;
R = np.zeros((H,L)) + obstacle_reward;
Q = np.random.rand(H,L)*1e-3;
#R = np.zeros((16,3))
#R states are 000, 001, 010, 011, 111, 110, 101, 100; 1=obstacle, 0=no obstacle
#R actions are move NE, move E, move SE

R[1,2] = obstacle_penalty;
R[2,1] = obstacle_penalty;
R[3,1:] = obstacle_penalty;
R[4,:] = obstacle_penalty;
R[5,:2] = obstacle_penalty;
R[6,[0,2]] = obstacle_penalty;
R[7,0] = obstacle_penalty;
epsilon = 0.1;
alpha = 0.1;
gamma = 0.6;

# ipdb.set_trace()


deltaQ_norm = 1;
Q_prev = Q*1.0;
st = random.randint(0,H-1);
iter=0;
tol=1e-8;
iter_min=500;
iter_max = 10000;
# while deltaQ_norm>tol:

while iter<iter_max:
    if deltaQ_norm<tol and iter>iter_min:
        break
    # ipdb.set_trace()
    if np.random.rand()<epsilon:
        ac = random.randint(0,L-1);
    else:
        ac = np.argmax(Q[st,:]);
    sprime = random.randint(0,H-1);
    Q[st,ac] = (1-alpha)*Q[st,ac] + alpha*(R[st,ac] + gamma*max(Q[sprime,:]));
    st = sprime;
    iter=iter+1;
    deltaQ = np.abs(Q-Q_prev);
    deltaQ_norm = np.mean(deltaQ);
    print("dQ = ", deltaQ_norm)
    Q_prev = Q*1.0;

Q_obstacle = Q;



#***************Litter module****************************

H=8; L=3;
# litter_penalty = -100;
litter_reward = 50.;
R = np.zeros((H,L));
Q = np.random.rand(H,L)*1e-3;
#R = np.zeros((16,3))
#R states are 000, 001, 010, 011, 111, 110, 101, 100; 1=litter, 0=no litter
#R actions are move NE, move E, move SE

R[1,2] = litter_reward;
R[2,1] = litter_reward;
R[3,1:] = litter_reward;
R[4,:] = litter_reward;
R[5,:2] = litter_reward;
R[6,[0,2]] = litter_reward;
R[7,0] = litter_reward;
epsilon = 0.1;
alpha = 0.2;
gamma = 0.5;

# ipdb.set_trace()


deltaQ_norm = 1;
Q_prev = Q*1.0;
st = random.randint(0,H-1);
iter=0;
tol=1e-3;
iter_min=5000;
iter_max = 10000;
# while deltaQ_norm>tol:
while iter<iter_max:
    if deltaQ_norm<tol and iter>iter_min:
        break
    # ipdb.set_trace()
    if np.random.rand()<epsilon:
        ac = random.randint(0,L-1);
    else:
        ac = np.argmax(Q[st,:]);
    sprime = random.randint(0,H-1);
    Q[st,ac] = (1-alpha)*Q[st,ac] + alpha*(R[st,ac] + gamma*max(Q[sprime,:]));
    st = sprime;
    iter=iter+1;
    deltaQ = np.abs(Q-Q_prev);
    deltaQ_norm = np.mean(deltaQ);
    print("dQ = ", deltaQ_norm)
    Q_prev = Q*1.0;

Q_litter = Q;



#***********************Generating map and executing agent***************************
H=7; L=25;
M = np.zeros((H,L));
start = random.randint(0,H-1);

O = np.random.randint(6, size=(H, L))
O[O!=1]=0;
j=0;
i=start;
# while j<L-1:



ipdb.set_trace()
