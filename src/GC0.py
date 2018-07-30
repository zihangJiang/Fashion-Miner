# -*- coding: utf-8 -*-
"""
Created on June 10 13:01:05 2018

@author: jzh

require:
cv2
numpy

usage:
1. use mouse to select foreground/background
2. click right mouse bottom to change mode of selecting background/foreground(or press '1')
3. press '4'/'5' to clear foreground/background mask(or press '2' to clear all)
4. press '3' to run grabcut(or press 'c'/'Enter')
5. press 'esc' to quit
"""

import numpy as np
import cv2

# 载入图片以及初始化
name_data = np.load('../data/image_used.npy')
img = cv2.imread(name_data[0])
# GrabCut所需内部调用的参数
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
iteration = 10
# 鼠标点击的时候变为True
drawing = False
# 如果为True, 选择前景。点击'm'或'1'或右键切换到背景选择
mode = True
# 记录鼠标点击时的位置
ix,iy = -1,-1
# 绘制点的半径
r = 8

# 将图片,以及选择的前景,背景一起显示出来
def merge(mask_fore, mask_back, img):
    mask = mask_fore[:,:,1:2]/255 + mask_back[:,:,2:3]/255
    if np.max(np.max(np.max(mask)))>1:
        return False, img
    mask = mask.astype('uint8')
    return True, mask_fore + mask_back + img*(1-mask)

# 鼠标回调函数
# 前景用绿色,背景用红色,不允许二者出现重叠!
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode,r
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    if event == cv2.EVENT_RBUTTONUP:
        mode = not mode

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            if mode is True:
                cv2.circle(mask_fore,(x,y),r,(0,255,0),-1)
            else:
                cv2.circle(mask_back,(x,y),r,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode is True:
            cv2.circle(mask_fore,(x,y),r,(0,255,0),-1)
        else:
            cv2.circle(mask_back,(x,y),r,(0,0,255),-1)

mask_fore = np.zeros_like(img)
mask_back = np.zeros_like(img)
black = np.load('../black.npy')
white = np.load('../white.npy')
mask_fore[:,:,1][white > 205] = 255
mask_back[:,:,2][black == 1] = 255   
mask_back[:270,-20:,2] = 255
mask_back[:270,:20,2] = 255
mask_back[:5,:,2] = 255       
for i in range(len(name_data)):
    # 初始化前景和背景的mask
    img = cv2.imread(name_data[i])

    
    valid, masked = merge(mask_fore, mask_back, img)
    
    
    
    print('Cutting foreground from the picture')

    mask_global = np.zeros(img.shape[:2],np.uint8) + 2
    mask_global[mask_fore[:,:,1] == 255] = 1
    mask_global[mask_back[:,:,2] == 255] = 0
    mask_global, bgdModel, fgdModel = cv2.grabCut(img,mask_global,None,bgdModel,fgdModel,iteration,cv2.GC_INIT_WITH_MASK)        
    mask_global = np.where((mask_global==2)|(mask_global==0),0,1).astype('uint8')
    target = img*mask_global[:,:,np.newaxis]
    cv2.imwrite(name_data[i],target)
            
    
