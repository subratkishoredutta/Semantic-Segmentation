# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 20:51:13 2021

@author: Asus
"""

def predict(path,model):##path = path to the file to be tested, model = trained model
  import cv2
  import keras
  import tensorflow as tf
  import numpy as np
  import matplotlib.pyplot as plt
  IMG_WIDTH = IMG_HEIGHT = 128
  img=cv2.imread(path,1)/255
  img=cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
  img=np.array([img],dtype=np.uint8)
  img=model.predict(img,verbose=1)
  pred = (img>0.5).astype(np.uint8)
  plt.imshow(np.squeeze(pred))
