# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:14:01 2018

@author: Administrator
"""

from src.CVAE import load_network
import numpy as np



img_array = np.load('data/whole_img_data_cut.npy')
encoder = load_network('model/2000_encoder.h5')
encode_data = encoder.predict(img_array/255)
np.save('data/encode_data_2000', encode_data)