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
keyboard.wait('Ctrl')
