# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:14:01 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.manifold
import sklearn.decomposition

# number of nodes to use
numnodes = 210
t = 5
n = 20
encode_data = np.load('../data/encode_data_2000.npy')[2]
#encode_data = np.load('../data/vgg_encode2.npy').reshape((210,40*512))
name_data = np.load('../data/image.npy')
#embed =sklearn.decomposition.PCA(n_components=2).fit_transform(encode_data)
embed =sklearn.manifold.TSNE(n_components=2).fit_transform(encode_data)
color = ['r.','ko','b*','yo','c*','ko','m.','r*','m*','b.','bo','k.','k*','y*','ro','bv','rv','kv','yv','mv','cv']
#for i in range(len(color)):
#    fig = plt.figure()
#    ax = plt.axes(projection='3d')
#    print(i)
#    ax.plot3D(embed[10*i:10*(i+1),0],embed[10*i:10*(i+1),1],embed[10*i:10*(i+1),2],color[i])
#    ax.plot3D(embed[:,0],embed[:,1],embed[:,2],'g.')
#
#    plt.show()
for i in range(len(color)):
    fig = plt.figure()

    print(i)
    plt.plot(embed[:,0],embed[:,1],'g.')
    plt.plot(embed[10*i:10*(i+1),0],embed[10*i:10*(i+1),1],color[i])


    plt.show()
