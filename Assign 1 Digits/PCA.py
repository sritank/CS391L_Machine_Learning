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

t = 20

V = v[:,0:t];
W = w[0:t];

train_img_V = np.matmul(train_img_2d, V);

correct_counter = 0;
total_counter = 0;
incorrect_test_index = np.array([]);
incorrect_train_index = np.array([]);

#test_no = 1008;
for test_no in range(0,10000):
    test_img_1 = test_img_2d[test_no];
    test_lab_1 = test_lab[test_no];
    total_counter = total_counter + 1

    #testImgArr = Image.fromarray(test_img_1.reshape(28,28));
    #testImgArr.show();

    test_img_1 = test_img_1 - col_sum;

    test_img_1_V = np.matmul(test_img_1,V);

    metro_dist_test = np.abs(test_img_1_V-train_img_V);
    #train_img_reduced = np.matmul(train_img_2d,W)
    T = np.sum(metro_dist_test, axis=1);

    i = np.argmin(T)

    out = train_lab[i];
    if (out == test_lab_1):
        correct_counter = correct_counter + 1
    else:
        # s = test_no
        # print("came here s=",s)
        incorrect_test_index = np.append(incorrect_test_index,[test_no]);
        incorrect_train_index = np.append(incorrect_train_index,[i]);

# print("k =   ",k, "   out=  ", out)
# print("ground truth value = ",test_lab_1)
print("total images", total_counter);
print("correctly identified images", correct_counter);

# for j in range(0,10):#enumerate(incorrect_train_index, start=0):
# j=5
# out_images = train_img[incorrect_train_index[j].astype(int)];
# incorrect_test_images = test_img[incorrect_test_index[j].astype(int)];
# inc_trainImg = Image.fromarray(out_images);
# inc_trainImg.show()
# inc_testImg = Image.fromarray(incorrect_test_images);
# inc_testImg.show();

#keyboard.wait('Ctrl')
