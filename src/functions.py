# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:14:01 2018

@author: Administrator
"""

import markov_clustering as mc
import networkx as nx
import random
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
def handle(name):
    start = name.find('RTW') + 4
    end = name.find('rtw') - 1
    return name[start : end]
    
def get_label(name_data, numnodes = None):
    if numnodes==None:
        numnodes = len(name_data)
    label = [handle(name)+'!'+str(index) for index,name in enumerate(name_data[:numnodes])]
    return label

def calculate_sim(encode_data):
    numnodes = len(encode_data)
    sim = np.zeros((numnodes,numnodes))
    for i in range(numnodes):
        for j in range(numnodes):
            if not i==j:
                dist = LA.norm(encode_data[i]-encode_data[j])
                if dist>5:
                    pass
                    #continue
                sim[i][j]=dist
    return sim

def export_edge_list_gephi(encode_data,matrix_form = False,thresh = 5, header = True, delim = ',', label = None, filename = 'edge_list.csv', cluster = None):
    f = open(filename, 'w')
    if header:
        f.write("Source,Target,Weight\n")
    if matrix_form:
        numnodes = encode_data.shape[0]
    else:
        numnodes = len(encode_data)
    if label == None:
        label = [str(i) for i in range(numnodes)]
    
    for i in range(numnodes):
        for j in range(numnodes):
            if not i==j:
                if matrix_form:
                    dist = encode_data[i][j]
                else:
                    dist = LA.norm(encode_data[i]-encode_data[j])
                if dist>thresh:
                    continue
                sent = "\"" + label[i] + "\"" + delim + "\"" + label[j] +"\""+ delim + str(dist) + "\n"
                f.write(sent)
    f.close()
    

    
def export_node_list_gephi(encode_data, matrix_form = False, thresh = 5,inflation = 1.5, header = True, delim = ',',label = None, filename = 'node_list.csv'):
    
    print('building graph')
    if matrix_form:
        numnodes = encode_data.shape[0]
        matrix = encode_data
    else:
        numnodes = len(encode_data)
        positions = {i:encode_data[i] for i in range(numnodes)}
        # use networkx to generate the graph
        network = nx.random_geometric_graph(numnodes, thresh, pos=positions)
        
        # then get the adjacency matrix (in sparse form)
        matrix = nx.to_scipy_sparse_matrix(network)
    #
    print('runing mcl')
    result = mc.run_mcl(matrix, inflation=inflation)
    clusters = mc.get_clusters(result)
    if label == None:
        label = [str(i) for i in range(numnodes)]
    f = open("node_list.csv", 'w')
    if header:
        f.write("Id,Label,Cluster-ID\n")
    i = 1
    for j in clusters:
        for node in j:
            pos = label[node].find('!')
            sent = "\"" + label[node] + "\"" + delim +"\"" + label[node][:pos] + "\"" + delim + str(i) +"\n"
            f.write(sent)
        i=i+1
    f.close()
    
def export_node_edge_list_gephi(encode_data,name_data, matrix_form = False, thresh = 3,inflation = 1.5, header = True, delim = ',',label = None, filename_edge = 'edge_list.csv',filename_node = 'node_list.csv'):
    
    print('building graph')
    if matrix_form:
        numnodes = encode_data.shape[0]
        matrix = encode_data
    else:
        numnodes = len(encode_data)
        positions = {i:encode_data[i] for i in range(numnodes)}
        # use networkx to generate the graph
        network = nx.random_geometric_graph(numnodes, thresh, pos=positions)
        
        # then get the adjacency matrix (in sparse form)
        matrix = nx.to_scipy_sparse_matrix(network)
    #
    print('runing mcl')
    result = mc.run_mcl(matrix, inflation=inflation)
    clusters = mc.get_clusters(result)
    if label == None:
        label = [str(i) for i in range(numnodes)]
    f = open(filename_node, 'w')
    if header:
        f.write("Id,Label,Cluster-ID,image\n")
    i = 1
    record_of_single = ()
    for j in clusters:
        if len(j)==1:
            record_of_single += j
            continue
        for node in j:
            pos = label[node].find('!')
            pos2 = name_data[node].find('RTW')
            sent = "\"" + label[node] + "\"" + delim +"\"" + label[node][:pos] + "\"" + delim + str(i) +delim+"\""+name_data[node][pos2+4:-4]+".png"+"\""+"\n"
            f.write(sent)
        i=i+1
    f.close()

    f = open(filename_edge, 'w')
    if header:
        f.write("Source,Target,Weight\n")
    if matrix_form:
        numnodes = encode_data.shape[0]
    else:
        numnodes = len(encode_data)
    if label == None:
        label = [str(i) for i in range(numnodes)]
    
    for i in range(numnodes):
        for j in range(numnodes):
            if (i in record_of_single) or (j in record_of_single):
                continue
            if not i==j:
                if matrix_form:
                    dist = encode_data[i][j]
                else:
                    dist = LA.norm(encode_data[i]-encode_data[j])
                if dist>thresh:
                    continue
                sent = "\"" + label[i] + "\"" + delim + "\"" + label[j] +"\""+ delim + str(dist) + "\n"
                f.write(sent)
    f.close()
    

    

    

def export_node_list_withimage(encode_data,name_data, matrix_form = False, thresh = 5,inflation = 1.5, header = True, delim = ',',label = None, filename = 'node_list.csv'):
    
    print('building graph')
    if matrix_form:
        numnodes = encode_data.shape[0]
        matrix = encode_data
    else:
        numnodes = len(encode_data)
        positions = {i:encode_data[i] for i in range(numnodes)}
        # use networkx to generate the graph
        network = nx.random_geometric_graph(numnodes, thresh, pos=positions)
        
        # then get the adjacency matrix (in sparse form)
        matrix = nx.to_scipy_sparse_matrix(network)
    
    print('runing mcl')
    #
    

    result = mc.run_mcl(matrix, inflation=inflation)
    clusters = mc.get_clusters(result)
    mc.draw_graph(matrix, clusters, node_size=50, with_labels=False, edge_color="silver")
    if label == None:
        label = [str(i) for i in range(numnodes)]
    f = open("node_list.csv", 'w')
    if header:
        f.write("Id,Label,Cluster-ID,image\n")
    i = 1
    for j in clusters:
        for node in j:
            pos = label[node].find('!')
            pos2 = name_data[node].find('RTW')
            sent = "\"" + label[node] + "\"" + delim +"\"" + label[node][:pos] + "\"" + delim + str(i) +delim+"\""+name_data[node][pos2+4:-4]+".png"+"\""+"\n"
            f.write(sent)
        i=i+1
    f.close()
def export_cluster_dirs(encode_data, name_data, matrix_form = False, thresh = 5,inflation = 1.5):
    
    print('building graph')
    if matrix_form:
        numnodes = encode_data.shape[0]
        matrix = encode_data
    else:
        numnodes = len(encode_data)
        positions = {i:encode_data[i] for i in range(numnodes)}
        # use networkx to generate the graph
        network = nx.random_geometric_graph(numnodes, thresh, pos=positions)
        
        # then get the adjacency matrix (in sparse form)
        matrix = nx.to_scipy_sparse_matrix(network)
    #
    print('runing mcl')
    #
    result = mc.run_mcl(matrix, inflation=inflation)
    clusters = mc.get_clusters(result)

    i = 1
    import shutil,os
    try:
        shutil.rmtree("cluster") 
    except:
        pass
    os.makedirs("cluster")
    for j in clusters:
        if len(j)==1:
            continue
        file = "cluster/{}".format(i)
        os.makedirs(file)
        for node in j:
            pos = name_data[node].find('RTW')
            shutil.copyfile(name_data[node],os.path.join(file,str(node)+name_data[node][pos+4:]))
        i=i+1

def draw_hist(sim):
    simflat = sim.reshape((-1,))
    #simflat = simflat[simflat != 1] # Too many ones result in a bad histogram so we remove them
    _ = plt.hist(simflat, bins=25)
    
    mmax  = np.max(simflat)
    mmin  = np.min(simflat)
    mmean = np.mean(simflat)
    print('avg={0:.2f} min={1:.2f} max={2:.2f}'.format(mmean, mmin, mmax))
    
def build_dataset(name_data,filename = 'whole_data'):
    import cv2
    image_array = cv2.imread(name_data[0])[np.newaxis,:]
    for name in name_data:
        if name == name_data[0]:
            continue
        img = cv2.imread(name)[np.newaxis,:]
        image_array = np.concatenate((image_array,img))
    np.save(filename,image_array)
    

#sumation = img_array[:,:,:,0]+img_array[:,:,:,1]+img_array[:,:,:,2]
#sumation[sumation>0]=1
#white = sum(sumation)
#white[white>0]=1
#black = 1-white
#white = sum(sumation)
#white[white<206] = 0