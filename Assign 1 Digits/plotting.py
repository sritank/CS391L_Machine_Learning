import matplotlib.pyplot as plt
import struct as st
import numpy as np
import keyboard
import idx2numpy
from PIL import Image
from numpy import linalg as LA
import pickle as pkl

data = pkl.load(open("plot_data.p","rb"));
eigV = data["eigV"];
accuracy = data["acc"]
print(eigV);
print();
l1, = plt.plot(eigV,accuracy, label='train size = 60000, test size = 500')
plt.ylabel('accuracy')
plt.xlabel('number of principal components')
plt.legend(handles = [l1])
plt.savefig('accuracy vs t.png')
