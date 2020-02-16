import matplotlib.pyplot as plt
import struct as st
import numpy as np
import keyboard
import idx2numpy
from PIL import Image
from numpy import linalg as LA
import pickle as pkl

# data = pkl.load(open("plot_data.p","rb"));
data = pkl.load(open("plot_data_train_size_10.p","rb"));
eigV = data["eigV"];
accuracy = data["acc"]
train = data["train"]
print(train);
print(accuracy);

data2 = pkl.load(open("plot_data_train_size_30.p","rb"));
eigV2 = data2["eigV"];
accuracy2 = data2["acc"]
train2 = data2["train"]
print(train2);
print(accuracy2);

l1, = plt.plot(train,accuracy*100, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4, label='PC = 10')
l2, = plt.plot(train2,accuracy2*100, marker='o', markerfacecolor='red', markersize=12, color='red', linewidth=4, label='PC = 30')
plt.ylabel('labeling accuracy in test dataset (%)')
plt.ylim([50, 100])
plt.xlabel('Size of training dataset (N)')
# plt.legend(handles = [l1])
plt.legend(handles = [l1,l2])
plt.savefig('accuracy vs train size.png')
plt.close()

###########Effect of PCs##################
data = pkl.load(open("plot_data_PCA_size_10000.p","rb"));
eigV = data["eigV"];
accuracy = data["acc"]
train = data["train"]
print(train);
print(accuracy);

data2 = pkl.load(open("plot_data_PCA_size_30000.p","rb"));
eigV2 = data2["eigV"];
accuracy2 = data2["acc"]
train2 = data2["train"]
print(train2);
print(accuracy2);

l1, = plt.plot(eigV,accuracy, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4, label='training set = 10000')
l2, = plt.plot(eigV,accuracy2, marker='o', markerfacecolor='red', markersize=12, color='red', linewidth=4, label='training set = 30000')
plt.ylabel('labeling accuracy (%)')
plt.xlabel('No. of principal components')
# plt.legend(handles = [l1])
plt.legend(handles = [l1,l2])
plt.savefig('accuracy vs no. of PCs.png')



############Effect of K#####################
