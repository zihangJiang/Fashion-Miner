# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:05:38 2018

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
import src.functions as ft


# number of nodes to use
numnodes = 3000
encode_data = np.load('data/encode_data_2000.npy')[2]
name_data = np.load('data/image_used.npy')

ft.export_cluster_dirs(encode_data[:numnodes],name_data,thresh = 2)

