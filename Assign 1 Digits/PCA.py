import struct as st
import numpy as np
import keyboard
import idx2numpy
from PIL import Image
from numpy import linalg as LA


train_img = idx2numpy.convert_from_file('train-images-idx3-ubyte');
test_img = idx2numpy.convert_from_file('t10k-images-idx3-ubyte');

#filename = {'images' : 'train-images-idx3-ubyte' ,'labels' : 'train-labels-idx1-ubyte'}
#train_imagesfile = open(filename['images'],'rb')
#trainImgArr = Image.fromarray(train_img[2],'L');
#trainImgArr.show();

train_img_2d = train_img.reshape(60000,28*28);
test_img_2d = test_img.reshape(10000,28*28);
col_sum = np.mean(train_img_2d, axis = 0);

train_img_2d = train_img_2d - col_sum;

train_lab = idx2numpy.convert_from_file('train-labels-idx1-ubyte');
test_lab = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte');

covariance_matrix = np.matmul(train_img_2d.transpose(),train_img_2d)/60000;

w,v = LA.eig(covariance_matrix);

<<<<<<< HEAD
t = 100
=======
t = 30
>>>>>>> 659a3928c1e927c7cf0b5c38df10d9b33b299a15
V = v[:,0:t];
W = w[0:t];

train_img_V = np.matmul(train_img_2d, V);

<<<<<<< HEAD
test_no = 900
=======
test_no = 1008
>>>>>>> 659a3928c1e927c7cf0b5c38df10d9b33b299a15
test_img_1 = test_img_2d[test_no];
test_lab_1 = test_lab[test_no];



testImgArr = Image.fromarray(test_img_1.reshape(28,28),'L');
testImgArr.show();

test_img_1 = test_img_1 - col_sum;

test_img_1_V = np.matmul(test_img_1,V);

metro_dist_test = np.abs(test_img_1_V-train_img_V);
#train_img_reduced = np.matmul(train_img_2d,W)
T = np.sum(metro_dist_test, axis=1);

k = np.argmin(T)

out = train_lab[k];
print("k =   ",k, "   out=  ", out)
print("ground truth value = ",test_lab_1)

out_image = train_img[k];
matchImg = Image.fromarray(out_image,'L');
matchImg.show()
#keyboard.wait('Ctrl')
