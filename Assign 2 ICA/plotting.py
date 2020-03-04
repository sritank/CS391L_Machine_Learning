import struct as st
import numpy as np
import keyboard
import idx2numpy
from PIL import Image
from numpy import linalg as LA
from numpy import matlib
import sounddevice as sd
import pickle as pkl
# import ipdb;
from scipy.io import loadmat;
from IPython.display import Audio
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
# plt.rcParams.update({'font.weight': 'bold'})

data = pkl.load(open("plot_data_ICA.p","rb"));
Uhat = data['Uhat']; U = data['U']; X = data['X']; What = data['What']; correl = data['correl'];

# A = ([0.5, 1, 0.2, 1, 0.3],[0.5, 0.5, 0.2, 0.6, 0.1],[0.8, 0.4, 0.5, 0.5, 0.31],[0.5, 0.4, 0.5, 1, 1],[0.2, 0.7, 0.7, 0.1, 0.3])

# ax = plt.axes()
#Plotting source signals
l1, = plt.plot(U[0,:], color='skyblue', linewidth=2, label = 'original signal 1')
l2, = plt.plot(U[1,:]+2.5, color='red', linewidth=2, label = 'original signal 2')
l3, = plt.plot(U[2,:]+5, color='green', linewidth=2, label = 'original signal 3')
l4, = plt.plot(U[3,:]+7.5, color='black', linewidth=2, label = 'original signal 4')
l5, = plt.plot(U[4,:]+10, color='yellow', linewidth=2, label = 'original signal 5')
plt.ylabel('signal level')
plt.ylim([-1.5,12])
plt.xlabel('time sample')
# plt.legend(handles = [l1,l2,l3,l4,l5])
plt.title('Original signals');# (y offset is for plotting purposes)')
plt.savefig('Original_signals.png')
plt.close();

#Plotting mixed signals
l1, = plt.plot(X[0,:], color='skyblue', linewidth=2, label = 'mixed signal 1')
l2, = plt.plot(X[1,:]+2.5, color='red', linewidth=2, label = 'mixed signal 2')
l3, = plt.plot(X[2,:]+5, color='green', linewidth=2, label = 'mixed signal 3')
l4, = plt.plot(X[3,:]+7.5, color='black', linewidth=2, label = 'mixed signal 4')
l5, = plt.plot(X[4,:]+10, color='yellow', linewidth=2, label = 'mixed signal 5')
plt.rcParams.update({'font.size': 14})
# plt.rcParams.update({'font.weight': 'bold'})
plt.ylabel('signal level')
plt.ylim([-1.5,12])
plt.xlabel('time sample')
# plt.legend(handles = [l1,l2,l3,l4,l5])
plt.title('Mixed signals');# (y offset is for plotting purposes)')
plt.savefig('Mixed_signals.png')
plt.close();

#Plotting retrieved signals
l1, = plt.plot(Uhat[0,:], color='skyblue', linewidth=2, label = 'retrieved signal 1')
l2, = plt.plot(Uhat[1,:]+35, color='red', linewidth=2, label = 'retrieved signal 2')
l3, = plt.plot(Uhat[2,:]+70, color='green', linewidth=2, label = 'retrieved signal 3')
l4, = plt.plot(Uhat[3,:]+105, color='black', linewidth=2, label = 'retrieved signal 4')
l5, = plt.plot(Uhat[4,:]+140, color='yellow', linewidth=2, label = 'retrieved signal 5')
plt.rcParams.update({'font.size': 14})
# plt.rcParams.update({'font.weight': 'bold'})
plt.ylabel('signal level')
plt.ylim([-20,155])
plt.xlabel('time sample')
# plt.legend(handles = [l1,l2,l3,l4,l5])
plt.title('Retrieved signals');# (y offset is for plotting purposes)')
plt.savefig('Retrieved_signals.png')
plt.close();

#Plotting Correlation between signals
plt.rcParams.update({'font.size': 10})
l1, = plt.plot(correl[:,0], color='skyblue', linewidth=2, label = 'correlation source 1')
l2, = plt.plot(correl[:,1], color='red', linewidth=2, label = 'correlation source 2')
l3, = plt.plot(correl[:,2], color='green', linewidth=2, label = 'correlation source 3')
l4, = plt.plot(correl[:,3], color='black', linewidth=2, label = 'correlation source 4')
l5, = plt.plot(correl[:,4], color='yellow', linewidth=2, label = 'correlation source 5')
# plt.rcParams.update({'font.weight': 'bold'})
plt.ylabel('Cross Correlation coefficient')
plt.ylim([-2,4])
plt.xlabel('iterations')
plt.legend(handles = [l1,l2,l3,l4,l5])
plt.title('Correlation between source and corresponding retrieved signals');# (y offset is for plotting purposes)')
plt.savefig('Correlation_signals.png')
plt.close();

#Plotting Homer simpson audio
plt.rcParams.update({'font.size': 14})
l1, = plt.plot(U[0,:], color='green', linewidth=4, label = 'original signal')
l2, = plt.plot(Uhat[2,:]/9.49, color='red', linewidth=1, label = 'retrieved scaled signal')
# plt.rcParams.update({'font.weight': 'bold'})
plt.ylabel('signal level')
plt.ylim([-1,2])
plt.xlim([21750,22250])
plt.legend(handles = [l1,l2])
plt.xlabel('time sample')
plt.title('Original vs retrieved scaled signal')
plt.savefig('Homer_Simpson_audio.png')
plt.close();
