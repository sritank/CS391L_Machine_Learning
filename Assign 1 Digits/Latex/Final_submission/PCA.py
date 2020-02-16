import struct as st
import numpy as np
import keyboard
import idx2numpy
from PIL import Image
from numpy import linalg as LA
import pickle as pkl

train_img = idx2numpy.convert_from_file('train-images-idx3-ubyte');
test_img = idx2numpy.convert_from_file('t10k-images-idx3-ubyte');

# train = np.array([500, 1000, 5000, 10000, 30000, 60000])
# PCA_size = np.array([5,10,20,30,50])
# K = np.array([1, 3, 5])

# train = np.array([500, 1000, 5000, 10000, 30000, 60000])
# PCA_size = np.array([30])
# k=1
# K = np.array([1, 3, 5])

train = np.array([30000])
PCA_size = np.array([5,10,20,30,50])
k=1

accuracy = [];
eigV = [];

for train_size in train:
    train_img_dup = train_img[0:train_size]
    train_img_2d = train_img_dup.reshape(train_size,28*28);
    test_img_2d = test_img.reshape(10000,28*28);
    col_sum = np.mean(train_img_2d, axis = 0);

    train_img_2d = train_img_2d - col_sum;

    train_lab = idx2numpy.convert_from_file('train-labels-idx1-ubyte');
    test_lab = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte');

    covariance_matrix = np.matmul(train_img_2d.transpose(),train_img_2d)/train_size;

    w,v = LA.eig(covariance_matrix);
    # accuracy = [];
    # eigV = [];

    ##################### Eigenvector and projection images section below. Uncomment to generate the images##############################
    # orig_image = train_img[0].reshape(28*28);
    # #######projecting onto first 30 principal components
    # proj = np.matmul(orig_image - col_sum,v[:,0:30])
    # proj_image = np.matmul(v[:,0:30],proj.transpose()) + col_sum; #projected image
    #
    #
    # projImg = Image.fromarray(proj_image.reshape(28,28));
    # projImg = projImg.convert('L');
    # projImg.save('Projected_figure.jpg');
    # origImg = Image.fromarray(orig_image.reshape(28,28),'L');
    # origImg.save('Original_figure.jpg');
    #
    #
    # eig_img0 = Image.fromarray(3e3*v[:,0].reshape(28,28));
    # eig_img0 = eig_img0.convert('L')
    # eig_img0.save('eig_IMG0.jpg')
    # eig_img1 = Image.fromarray(3e3*v[:,1].reshape(28,28));
    # eig_img1 = eig_img1.convert('L')
    # eig_img1.save('eig_IMG1.jpg')
    # eig_img2 = Image.fromarray(3e3*v[:,2].reshape(28,28));
    # eig_img2 = eig_img2.convert('L')
    # eig_img2.save('eig_IMG2.jpg')
    ##########################################################################################################################################

    for t in PCA_size: #first n eigen vectors

        V = v[:,0:t];
        W = w[0:t];

        train_img_V = np.matmul(train_img_2d, V);

        correct_counter = 0;
        # total_counter = 0;
        incorrect_test_index = np.array([]);
        incorrect_train_index = np.array([]);


        for test_no in range(0,10000): #accuracy across 10000 test images
            test_img_1 = test_img_2d[test_no];
            test_lab_1 = test_lab[test_no];
            # total_counter = total_counter + 1

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
            # else:
                # s = test_no
                # print("came here s=",s)
                # incorrect_test_index = np.append(incorrect_test_index,[test_no]);
                # incorrect_train_index = np.append(incorrect_train_index,[i]);

        # print("k =   ",k, "   out=  ", out)
        # print("ground truth value = ",test_lab_1)
        print("total test images", 10000);
        print('total components', t);
        print('training set size', train_size);
        print('k value', 1)
        print("correctly identified images", correct_counter);
        # eigV = np.append(eigV,[t]);
        accuracy = np.append(accuracy,[correct_counter/10000]);
        print(accuracy)
        # print
# data = {"train": train, "eigV":t, "acc":accuracy, "K":k}
# pkl.dump(data,open("plot_data_train_size_30.p","wb"));

data = {"train": train, "eigV":PCA_size, "acc":accuracy, "K":k}
pkl.dump(data,open("plot_data_PCA_size_30000.p","wb"));

    # data = {"train": train_size, "eigV":eigV, "acc":accuracy}
    # pkl.dump(data,open("plot_data.p","wb"));

# for j in range(0,10):#enumerate(incorrect_train_index, start=0):
# j=5
# out_images = train_img[incorrect_train_index[j].astype(int)];
# incorrect_test_images = test_img[incorrect_test_index[j].astype(int)];
# inc_trainImg = Image.fromarray(out_images);
# inc_trainImg.show()
# inc_testImg = Image.fromarray(incorrect_test_images);
# inc_testImg.show();

#keyboard.wait('Ctrl')
