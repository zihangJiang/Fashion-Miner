# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:05:38 2018

@author: Administrator
"""
import numpy as np
import cv2
import src.functions as ft
from keras.preprocessing.image import array_to_img


name_data = np.load('data/image.npy')
rename = ft.get_label(name_data)
coresponder = {rename[i]:name_data[i] for i in range(len(name_data))}

name = 'annasui!63'
image = cv2.cvtColor(cv2.imread(coresponder[name]),cv2.COLOR_BGR2RGB)

image = array_to_img(image)
