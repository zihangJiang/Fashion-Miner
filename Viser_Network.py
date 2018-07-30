# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 13:47:18 2018

@author: Administrator
"""

from src.CVAE import load_network
import numpy as np

encoder = load_network('model/2000_encoder.h5')
decoder = load_network('model/2000_decoder.h5')
from keras.preprocessing.image import array_to_img


[array_to_img(encoder.get_weights()[0][:,:,1:2,i]).resize((100,100)).save("net_kernel/{}.png".format(i), quality=100) for i in range(8)]