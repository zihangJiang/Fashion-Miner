# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:14:01 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import src.functions as ft


# number of nodes to use
numnodes = 3000
t = 4
n = 20
encode_data = np.load('data/encode_data_2000.npy')[2,:,:n]
name_data = np.load('data/image_used.npy')
origin_data = name_data
name_data = ft.get_label(name_data)

#ft.export_edge_list_gephi(encode_data[:numnodes],matrix_form = False,label = name_data,thresh = t)
#ft.export_node_list_withimage(encode_data[:numnodes],origin_data,matrix_form = False,label = name_data,thresh = t)
ft.export_node_edge_list_gephi(encode_data[:numnodes],origin_data,matrix_form = False,label = name_data,thresh = t)