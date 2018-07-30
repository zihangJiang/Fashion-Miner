# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:05:38 2018

@author: Administrator
"""
from src.CVAE import load_network
import numpy as np
import cv2
from keras.preprocessing.image import array_to_img

img_array = np.load('data/whole_img_data.npy')
encoder = load_network('model/20_encoder.h5')
decoder = load_network('model/20_decoder.h5')
encode_data = encoder.predict(img_array/255)
a = encode_data[2].mean(axis = 0)
#a = encode_data[2][50:51]
#b = encode_data[2][2:3]
#c = encode_data[2][32:33]
#[array_to_img(cv2.cvtColor((decoder.predict((a*i+b*(6-i))*(1/6))*255)[0],cv2.COLOR_BGR2RGB)).save("gif/a-b{}.png".format(i), quality=100) for i in range(7)]
#
#[array_to_img(cv2.cvtColor((decoder.predict((a*i+c*(6-i))*(1/6))*255)[0],cv2.COLOR_BGR2RGB)).save("gif/a-c{}.png".format(i), quality=100) for i in range(7)]
#
#[array_to_img(cv2.cvtColor((decoder.predict((b*i+c*(8-i))*(1/8))*255)[0],cv2.COLOR_BGR2RGB)).save("gif/b-c{}.png".format(i), quality=100) for i in range(9)]
