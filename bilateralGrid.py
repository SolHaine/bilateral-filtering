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


def gaussianFilter(sigmaS, sigmaR, gamma) :
    #convolve on the 3 axis
    N = int(3 * (sigmaS + sigmaR))
    x= np.arange(-N,N+1,1) 
    y = x 
    i = x
    xx, yy, ii = np.meshgrid(x,y,i)
    kernel = np.exp(-((xx**2 + yy**2/(2*sigmaS**2)) + ((ii**2)/(2*sigmaR**2))))
    return kernel

def smoothGaussian(gamma, sigmaS, sigmaR):
    kernel = gaussianFilter(sigmaS, sigmaR, gamma)
    print('kernel shape ', kernel.shape)
    gamma2=scipy.ndimage.convolve(gamma.astype(float), kernel)
    #gamma_smooth= scipy.signal.convolve(gamma2.astype(float), kernel)  
    gamma_smooth= scipy.ndimage.convolve(gamma2.astype(float), np.array([kernel[0].T,kernel[1].T]))
    return gamma_smooth

#Wi get gamma : 3D structure
#W get non negative weight
def get_Wi_W(X,Y, I, sigS, sigR, R) :
    Wi = np.zeros((X,Y,R+1))
    print('Wi shape in downS', Wi.shape)
    W = np.zeros((X,Y,int(np.round(R))+1))
    for i in range(X) :
        for j in range(Y) :
            z = int(np.round(I[i,j]*R))
            assert(z>=0 and z<R+1)
            #Gamma 3D
            #Intensity
            #print('z ',z)
            Wi[i,j,z] = I[i,j]
            #Weight => To count how many pixel has the same intensity
            W[i,j,z] +=1
    return Wi,W


#Pseudo code source

#without down and up sampling
# http://people.csail.mit.edu/sparis/bf_course/course_notes.pdf

#with down and up sampling
#http://people.csail.mit.edu/sparis/publi/2009/fntcgv/Paris_09_Bilateral_filtering.pdf

def bilateralGrid(sigmaS, sigmaR, image):
    #intensity of one pixel = gray level 
    I = skimage.color.rgb2gray(image)
    plt.imshow(I,cmap=plt.cm.Greys_r)
    plt.pause(1)
    X = image.shape[0]
    Y = image.shape[1]
    R = 20
    Wi, W = get_Wi_W(X,Y,I,sigmaS,sigmaR,R)
    print('Wi shape', Wi.shape)
    print('W shape', W.shape)
    Wi_smooth = smoothGaussian(Wi, sigmaS, sigmaR)
    W_smooth = smoothGaussian(W, sigmaS, sigmaR)
    print('Wi_smooth shape', Wi_smooth.shape)
    
    newX = Wi_smooth.shape[0]
    newY = Wi_smooth.shape[1]
    print('BF X', newX)
    print('BF Y', newY)
    BF = np.zeros((newX,newY)).astype(float)
    print('bf shape', BF.shape)
    for i in range(Wi_smooth.shape[0]) :
        for j in range(Wi_smooth.shape[1]) :
            #print('np.sum(Wi_smooth[i,j])', np.sum(Wi_smooth[i,j]))
            #print('np.sum(W_smooth[i,j])', np.sum(W_smooth[i,j]))
            #There is normally one element != 0, so we make the sum to get it
            # wI~ / w~
            z = int(np.round(I[i,j]*R))
            BF[i,j] += abs( Wi_smooth[i,j,z]/ W_smooth[i,j,z])
            if (BF[i,j] > 1) :
                BF[i,j] =1
    return BF

def main() : 
    im=skimage.io.imread('ghibli.jpg')
    startTime=time.time()
    BF = bilateralGrid(16,0.8,im)
    endTime =time.time()
    duration = endTime - startTime
    if(duration < 60) :
        print('execution time - complexity', duration, "secondes" )
    else :
        print('execution time - complexity', duration, "secondes" )
        print('execution time - complexity', int(duration/60), " minutes", abs( int(duration/60) - duration/60 )*60, 'secondes' )
    plt.imshow(BF,cmap=plt.cm.Greys_r)
    skimage.io.imsave('./bilateral_grid_outputs/BG_ghibli_16_08.png', BF)
    

main()
