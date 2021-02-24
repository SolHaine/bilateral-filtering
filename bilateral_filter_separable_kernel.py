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
                        #print("pixel",i, j, "and", x,j) #so we can see where we are
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
                    #print("pixel",i, j, "and", i,y) #so we can see where we are
                    w = gaussianKernel(norm, sigmaS) * gaussianKernel(np.abs(image[i,j]-image[i,y]), sigmaR)
                    bf_image[i,j] += w*image[i,y]
                    Wp += w
            bf_image[i,j] = bf_image[i,j] / Wp
    return bf_image


def main():
    imageName ='ghibli.jpg'
    image= imageio.imread(imageName)
    image = image.astype(np.float32)/255.0
        
    #§plt.imshow(im,cmap=plt.cm.Greys_r)
    #plt.pause(1)
    
    startTime=time.time()
    
    sigmaS = 16
    sigmaR = 0.1
    
    red_BF_row = bruteForceRow(image[:,:,0], sigmaS, sigmaR)
    red_BF = bruteForceCol(red_BF_row, sigmaS, sigmaR)    
    
    green_BF_row = bruteForceRow(image[:,:,1], sigmaS, sigmaR)
    green_BF = bruteForceCol(green_BF_row, sigmaS, sigmaR)    
    
    blue_BF_row = bruteForceRow(image[:,:,2], sigmaS, sigmaR)
    blue_BF = bruteForceCol(blue_BF_row, sigmaS, sigmaR)    
    
    BF = np.stack([red_BF, green_BF, blue_BF], axis=2)
    endTime =time.time()
    
    duration = endTime - startTime
    if(duration < 60) :
        print('execution time - complexity', duration, "secondes" )
    else :
        print('execution time - complexity', duration, "secondes" )
        print('execution time - complexity', int(duration/60), " minutes", abs( int(duration/60) - duration/60 )*60, 'secondes' )
    #
    plt.imshow(BF,cmap=plt.cm.Greys_r)
    #imageio.imwrite("./Outputs/SeparateKernel" + imageName + "sigmaS_" + str( sigmaS) + "_sigmaR_" + "0.05" +  ".png", im)
    # stack the images horizontally
    #skimage.io.imsave('./separable_kernel_outputs/Ghibli/Color/SK_ghibli_1_08.png', BF)
    #skimage.io.imsave('./separable_kernel_outputs/Ghibli/Color/SK_ghibli_2_08.png', BF)
    #skimage.io.imsave('./separable_kernel_outputs/Ghibli/Color/SK_ghibli_4_005.png', BF)
    #skimage.io.imsave('./separable_kernel_outputs/Ghibli/Color/SK_ghibli_8_08.png', BF)
    skimage.io.imsave('./separable_kernel_outputs/Ghibli/Color/SK_ghibli_16_01.png', BF)
    
main()