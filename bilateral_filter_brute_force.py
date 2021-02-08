# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 15:37:46 2021

@author: Solène
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy import ndimage
import imageio
import skimage.color

def gaussianKernel(x, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x*x) / (2*sigma*sigma))
    

def bruteForceBilateralFilter(image, sigmaS, sigmaR):
    bf_image = np.zeros_like(image);
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            Wp = 0
            for x in range(image.shape[0]):
                for y in range(image.shape[1]):
                    norm = np.sqrt((i-x)*(i-x)+(j-y)*(j-y))
                    if norm < 2*sigmaS:
                        print("pixel",i, j, "and", x,y) #so we can see where we are
                        w = gaussianKernel(norm, sigmaS) * gaussianKernel(np.abs(image[i,j]-image[x,y]), sigmaR)
                        bf_image[i,j] += w*image[x,y]
                        Wp += w
            bf_image[i,j] = bf_image[i,j] / Wp
    
    return bf_image


def main():
    image_path = "cat_small.png"
    image = np.array(imageio.imread(image_path))
    image_grayscale = skimage.color.rgb2gray(image)
    plt.imshow(image_grayscale, cmap=plt.cm.gray)
    print(image_grayscale)
    
    filtered_image = bruteForceBilateralFilter(image_grayscale, 4, 0.2) #takes a while with sigmaS > 2
    
    plt.imshow(filtered_image, cmap=plt.cm.gray)
    
    imageio.imwrite("./brute_force_outputs/filtered_cat_4_02.png", filtered_image)
    
    

if __name__ == "__main__":
    main()