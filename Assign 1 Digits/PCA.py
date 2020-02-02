import struct as st
import numpy as np
import keyboard
import idx2numpy
from PIL import Image
train_img = idx2numpy.convert_from_file('train-images-idx3-ubyte')

#filename = {'images' : 'train-images-idx3-ubyte' ,'labels' : 'train-labels-idx1-ubyte'}
#train_imagesfile = open(filename['images'],'rb')
trainImgArr = Image.fromarray(train_img[2],'L');
trainImgArr.show();

train_img_2d = train_img.reshape(60000,28*28)

train_lab = idx2numpy.convert_from_file('train-labels-idx1-ubyte')

#covariance_matrix = np.matmul(train_lab.transpose(),train_lab)



#keyboard.wait('Ctrl')
