import math
import os
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.filters import threshold_otsu, threshold_isodata
from skimage import exposure

import Mask
import ImageIO

center = (256, 252)

# Takes in an image as an np ndarray.
# Name and date are for saving purposes
# TODO Refactor to return image, save elsewhere
def six_cloud_contrast(img, name, date):
    # Find the mask and black out those pixels.
    mask = Mask.find_mask()
    img = Mask.apply_mask(mask, img)
    
    # This blocks out all the stars, helping the local threshold later be more
    # accurate.
    img2 = ndimage.grey_opening(img, size=(2,2))
    
    p1, p2 = np.percentile(img2, (5, 95))
    imgEq2 = exposure.rescale_intensity(img2, in_range=(p1, p2))
    
    #imgEq2 = exposure.equalize_hist(img2)
    
    if not np.array_equal(imgEq2, np.zeros(imgEq2.shape)):
        img2 = np.copy(imgEq2)
    
    imgT = np.copy(img2)
    half = imgT.shape[0] // 2
    topleft = imgT[0:half, 0:half]
    bottomleft = imgT[half:img2.shape[1], 0:half]
    topright = imgT[0:half, half:img2.shape[0]]
    bottomright = imgT[half:img2.shape[1], half:img2.shape[0]]
    
    thresh = threshold_isodata(topleft)
    topleft2 = np.where(topleft > thresh, 255, 0)
    thresh = threshold_isodata(topright)
    topright2 = np.where(topright > thresh, 255, 0)
    thresh = threshold_isodata(bottomleft)
    bottomleft2 = np.where(bottomleft > thresh, 255, 0)
    thresh = threshold_isodata(bottomright)
    bottomright2 = np.where(bottomright > thresh, 255, 0)
    
    imgT = np.zeros(img2.shape)
    imgT[0:half, 0:half] = topleft2
    imgT[half:img2.shape[1], 0:half] = bottomleft2
    imgT[0:half, half:img2.shape[0]] = topright2
    imgT[half:img2.shape[1], half:img2.shape[0]] = bottomright2

    #imgT = ndimage.binary_closing(imgT, [[True, True]])
    
    # Blocks over any speckling that occurs in the local threshold.
    img3 = ndimage.binary_closing(imgT)#, [[True, True]])
    
    img3 = ndimage.binary_erosion(img3, [[True, True, True]])
    img3 = ndimage.binary_erosion(img3, [[False, True, False],[False, True, False],[False, True, False]])
    
    temp = np.copy(img)
    for row in range(0,img3.shape[0]):
        for column in range(0,img3.shape[1]):
        
            x = column - center[0]
            y = center[1] - row
            r = math.sqrt(x**2 + y**2)
            if r <= 240 and img3[row, column] == 0:# and temp[row, column] < 44:
                #print(temp[row, column])
                temp[row, column] = 0#temp[row, column] * .5#- 40
            
                if temp[row, column] < 0:
                    temp[row, column] = 0
    
    img4 = np.uint8(temp)

    loc = 'Images/Cloud/' + str(date)

    ImageIO.save_image(img4, name, loc, 'gray')



# Takes in an image as an np ndarray.
# Name and date are for saving purposes
def six_cloud_contrast2(img, name, date):
    
    # Find the mask and black out those pixels.
    mask = Mask.find_mask()
    img = Mask.apply_mask(mask, img)

    # Closing
    img2 = ndimage.grey_closing(img, size=(2,2))

    # Inverts
    img3 = np.subtract(255, img2)

    # Subtract closing from the invert to increase contrast. 
    # If it goes negative I want it to be 0 rather than positive abs of num.
    temp = np.int16(img3) - np.int16(img2)
    cond = np.less(temp, 0)
    img4 = np.where(cond, 0, temp)

    # Subtract the original from the working image.
    # Increases contrast with clouds.
    temp = np.int16(img4) - np.int16(img)
    cond = np.less(temp, 0)
    img7 = np.where(cond, 0, temp)
    
    thresh = threshold_otsu(img7)
    #print(thresh)
    
    # Threshold the image so we have the clouds white and everything else black.
    img6 = np.where(img7 > thresh, 255, 0)

    temp = np.int16(img)
    
    for row in range(0,img6.shape[0]):
        for column in range(0,img4.shape[1]):
        
            x = column - center[0]
            y = center[1] - row
            r = math.sqrt(x**2 + y**2)
            if r <= 240 and img6[row, column] == 255:# and temp[row, column] < 44:
                    #print(temp[row, column])
                temp[row, column] = 0#temp[row, column] *.5#- 40
            
                if temp[row, column] < 0:
                    temp[row, column] = 0

    img8 = np.uint8(temp)

    loc = 'Images/Cloud/' + str(date)

    ImageIO.save_image(img8, name, loc, 'gray')
    
date = '20170624'
directory = 'Images/Original/' + date + '/'
files = os.listdir(directory)
#file = 'r_ut071723s05040.png'
for file in files:
    img = ndimage.imread(directory + file, mode='L')
    six_cloud_contrast(img, file, date)
    

