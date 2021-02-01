# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:22:43 2021

@author: Asus
"""
import cv2
import matplotlib.pyplot as plt


img=cv2.imread('img.jpg')

img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img2=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
img3=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(img1)
plt.imshow(img2)
plt.imshow(img3)




