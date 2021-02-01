# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 09:56:42 2020

@author: Asus
"""
from tqdm import tqdm
import os
import cv2
img = cv2.imread('test.jpg')
'''
newimage1 = cv2.flip(img,0)##vertical
newimage2= cv2.flip(img,1)##hprizontal flip
newimage3 = cv2.flip(img,-1)#
'''
inputPATH='E:/rognidaan internship/SUBRATA/sub/'
outputPATH='E:/rognidaan internship/SUBRATA/mask/'

destI='E:/rognidaan internship/SUBRATA/AUG/input/'
destO='E:/rognidaan internship/SUBRATA/AUG/mask/'

def generate(outputPATH,destI):
    for name in tqdm(os.listdir(outputPATH)):
        img = cv2.imread(outputPATH+name)
        newimage1 = cv2.flip(img,0)##vertical
        newimage2= cv2.flip(img,1)##hprizontal flip
        newimage3 = cv2.flip(img,-1)#horizontal+vertical flip
        destination = destI + name
        cv2.imwrite(destination +'_1.jpg',img)
        cv2.imwrite(destination +'_2.jpg',newimage1)
        cv2.imwrite(destination +'_3.jpg',newimage2)
        cv2.imwrite(destination +'_4.jpg',newimage3)
    
    
generate(inputPATH,destI)    
generate(outputPATH,destO)
