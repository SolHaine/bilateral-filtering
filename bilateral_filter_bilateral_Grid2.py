# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 17:03:44 2021


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

import time


#3D kernel filter / Bilateral grid
#http://people.csail.mit.edu/sparis/bf_course/slides/06_implementation.pdf
#http://people.csail.mit.edu/sparis/publi/2009/fntcgv/Paris_09_Bilateral_filtering.pdf


def gaussianKernel(x, sigma):
    return  np.exp(-(x**2) / (2*(sigma**2) ) )
    
#Wi get gamma : 3D structure
#W get non negative weight
def get_Gamma(im) :
    W = im.shape[0]
    H = im.shape[1]
    I = 256
    gamma = np.zeros((W,H,I,2))
    
    for i in range(W) :
        for j in range(H) :
            k = int(np.round(im[i,j]*255))
            gamma[i,j,k] = [im[i,j], 1]
    print('end gamma')
    return gamma

def gaussianOnGamma(gamma, sigmaS, sigmaR) :
    W = gamma.shape[0]
    H = gamma.shape[1]
    I = gamma.shape[2]
    GB_gamma = np.zeros_like(gamma)
    for i in range(W) :
        for j in range(H) :
            for k in range(I):
                GB_gamma[i,j,k] = [ gaussianKernel(i, sigmaS) * gaussianKernel(j, sigmaS) * gaussianKernel(gamma[i,j,k][0], sigmaR),
                                   gaussianKernel(i, sigmaS) * gaussianKernel(j, sigmaS) * gaussianKernel(gamma[i,j,k][1], sigmaR)]
                #print('GB_gamma[i,j,k]', GB_gamma[i,j,k])
    print('end GB gamma')
    return GB_gamma
                

#Pseudo code source

#without down and up sampling
# http://people.csail.mit.edu/sparis/bf_course/course_notes.pdf

#with down and up sampling
#http://people.csail.mit.edu/sparis/publi/2009/fntcgv/Paris_09_Bilateral_filtering.pdf

def bilateralGrid(sigmaS, sigmaR, image):
    #assert(sigmaS>=1 and sigmaR>=1)
    #intensity of one pixel = gray level 
    #I = skimage.color.rgb2gray(image)
    I = skimage.color.rgb2gray(skimage.color.rgba2rgb(image))
    plt.imshow(I,cmap=plt.cm.Greys_r)
    plt.pause(1)
    
    gamma = get_Gamma(I)
    GB_gamma = gaussianOnGamma(gamma, sigmaS, sigmaR)
    
    BF = np.zeros_like(I)
    W = GB_gamma.shape[0]
    H = GB_gamma.shape[1]
    
    for i in range (BF.shape[0]) :
        wi =0
        w=0
        for j in range (BF.shape[1]) :
            k = int(np.round(I[i,j]*255)) 
            wi = GB_gamma[i,j,k][0]
            print('wi', GB_gamma[i,j,k][0])
            w = GB_gamma[i,j,k][1]
            print('w', GB_gamma[i,j,k][1])
            BF[i,j] = abs(wi/w)
            if (BF[i,j] > 1) :
                BF[i,j] = 1
    return BF

def main() : 
    im=skimage.io.imread('cat_small.png')
    startTime=time.time()
    BF = bilateralGrid(4,0.8,im)
    endTime =time.time()
    print('execution time - complexity', (endTime - startTime  )/60, " minutes")
    plt.imshow(BF,cmap=plt.cm.Greys_r)
    

main()
