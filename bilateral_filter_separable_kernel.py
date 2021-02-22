# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 16:15:41 2021

@author: linej
"""

import numpy as np
import scipy.signal as sig
import scipy.ndimage
import skimage.io # import de la librairie scikit-image
import skimage.exposure
from skimage.color import rgb2gray
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import imageio

import time


def gaussianKernel(x, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x*x) / (2*sigma*sigma))
    

def bruteForceCol(image, sigmaS, sigmaR):
    bf_image = np.zeros_like(image);
    W = image.shape[0]
    H = image.shape[1]
    
    for i in range(W):
        for j in range(H):
            Wp = 0
            for x in range(W):
                    norm = np.sqrt((i-x)*(i-x))
                    if norm < 2*sigmaS:
                        print("pixel",i, j, "and", x,j) #so we can see where we are
                        w = gaussianKernel(norm, sigmaS) * gaussianKernel(np.abs(image[i,j]-image[x,j]), sigmaR)
                        bf_image[i,j] += w*image[x,j]
                        Wp += w
            bf_image[i,j] = bf_image[i,j] / Wp
    
    return bf_image

def bruteForceRow(image, sigmaS, sigmaR):
    bf_image = np.zeros_like(image);
    W = image.shape[0]
    H = image.shape[1]
    
    for i in range(W):
        for j in range(H):
            Wp = 0
            for y in range(H):
                norm = np.sqrt((j-y)*(j-y))
                if norm < 2*sigmaS:
                    print("pixel",i, j, "and", i,y) #so we can see where we are
                    w = gaussianKernel(norm, sigmaS) * gaussianKernel(np.abs(image[i,j]-image[i,y]), sigmaR)
                    bf_image[i,j] += w*image[i,y]
                    Wp += w
            bf_image[i,j] = bf_image[i,j] / Wp
    return bf_image


def main():
    imageName ='cat_small.png'
    image=skimage.io.imread(imageName)
    im = skimage.color.rgb2gray(image)
    startTime=time.time()
    sigmaS = 4
    sigmaR = 0.05
    BF1 = bruteForceRow(im, sigmaS, sigmaR)
    BF = bruteForceCol(BF1, sigmaS, sigmaR)
    endTime =time.time()
    print('execution time - complexity', (endTime - startTime  )/60, " minutes" )
    #6 min cat small 4,0.8
    #22,9 min chihiro 4,0.8
    #6,58 min cat small 4,0.2
    #6,51 min cat small 4,0.05
    
    plt.imshow(BF,cmap=plt.cm.Greys_r)
    #imageio.imwrite("./Outputs/SeparateKernel" + imageName + "sigmaS_" + str( sigmaS) + "_sigmaR_" + "0.05" +  ".png", im)
    
main()