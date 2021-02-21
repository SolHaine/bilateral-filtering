# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 15:37:46 2021

@author: Solène
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import skimage.color
import time

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
    image_path = "ghibli.jpg"
    image = np.array(imageio.imread(image_path))
    image_grayscale = skimage.color.rgb2gray(image)
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image_grayscale, cmap=plt.cm.gray)
    print(image_grayscale)
    
    start_time = time.time()
    filtered_image = bruteForceBilateralFilter(image_grayscale, 2, 0.2) #takes a while with sigmaS > 2 or big images
    print("--- %s seconds ---" % (time.time() - start_time))
    
    f.add_subplot(1,2, 2)
    plt.imshow(filtered_image, cmap=plt.cm.gray)
    
    imageio.imwrite("./brute_force_outputs/ghibli_test_2_02.png", filtered_image)
    
    

if __name__ == "__main__":
    main()