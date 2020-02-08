import struct as st
import numpy as np
import keyboard
import idx2numpy
from PIL import Image
from numpy import linalg as LA
import pickle as pkl

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
accuracy = [];
eigV = [];
# orig_image=np.ones(56,56)
orig_image=[]
proj_image=[]
# proj_image=np.ones(56,56)
#Image before projection
# for i in range(0,3):
    # orig_image = np.append(orig_image, [train_img[i].reshape(28*28)]);
orig_image = train_img[0].reshape(28*28);
#projecting onto first 25 principal components
proj = np.matmul(orig_image - col_sum,v[:,0:30])
# proj_image = np.append(proj_image, [np.matmul(v[:,0:100],proj.transpose()) + col_sum]); #projected image
proj_image = np.matmul(v[:,0:30],proj.transpose()) + col_sum; #projected image

# Proj_image[0:27,:] = [proj_image[0].reshape(28,28), proj_image[1].reshape(28,28)]
# Proj_image[28:55,:] = [proj_image[2].reshape(28,28), proj_image[3].reshape(28,28)]
#
# Orig_image[0:27,:] = [orig_image[0].reshape(28,28), orig_image[1].reshape(28,28)]
# Orig_image[28:55,:] = [orig_image[2].reshape(28,28), orig_image[3].reshape(28,28)]

# Proj_image[0:27,:] = [proj_image[0].reshape(28,28), proj_image[1].reshape(28,28)]

# Proj_image = np.concatenate((proj_image[0:27,0:55],proj_image[0:27,56:111]),axis=0)
# Proj_image[28:55,:] = [proj_image[2].reshape(28,28), proj_image[3].reshape(28,28)]

# Orig_image[0:27,:] = [orig_image[0].reshape(28,28), orig_image[1].reshape(28,28)]
# Orig_image[28:55,:] = [orig_image[2].reshape(28,28), orig_image[3].reshape(28,28)]

projImg = Image.fromarray(proj_image.reshape(28,28));
projImg = projImg.convert('L');
projImg.save('Projected_figure.jpg');
origImg = Image.fromarray(orig_image.reshape(28,28),'L');
origImg.save('Original_figure.jpg');


eig_img0 = Image.fromarray(3e3*v[:,0].reshape(28,28));
eig_img0 = eig_img0.convert('L')
eig_img0.save('eig_IMG0.jpg')
eig_img1 = Image.fromarray(3e3*v[:,1].reshape(28,28));
eig_img1 = eig_img1.convert('L')
eig_img1.save('eig_IMG1.jpg')
eig_img2 = Image.fromarray(3e3*v[:,2].reshape(28,28));
eig_img2 = eig_img2.convert('L')
eig_img2.save('eig_IMG2.jpg')


for t in range(1,2):

    V = v[:,0:t];
    W = w[0:t];

    train_img_V = np.matmul(train_img_2d, V);

    correct_counter = 0;
    total_counter = 0;
    incorrect_test_index = np.array([]);
    incorrect_train_index = np.array([]);

    #test_no = 1008;
    for test_no in range(0,500):
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
    print('total components',t)
    print("correctly identified images", correct_counter);
    eigV = np.append(eigV,[t]);
    accuracy = np.append(accuracy,[correct_counter/total_counter]);

data = {"eigV":eigV, "acc":accuracy}
pkl.dump(data,open("plot_data.p","wb"));

# for j in range(0,10):#enumerate(incorrect_train_index, start=0):
# j=5
# out_images = train_img[incorrect_train_index[j].astype(int)];
# incorrect_test_images = test_img[incorrect_test_index[j].astype(int)];
# inc_trainImg = Image.fromarray(out_images);
# inc_trainImg.show()
# inc_testImg = Image.fromarray(incorrect_test_images);
# inc_testImg.show();

#keyboard.wait('Ctrl')
