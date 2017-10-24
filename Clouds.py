import math
import os
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.filters import threshold_otsu, threshold_isodata, threshold_li, threshold_local
from skimage import exposure

from matplotlib.patches import Circle

import Mask
import ImageIO

center = (256, 252)

# Takes in an image as an np ndarray.
# Name and date are for saving purposes
# TODO Refactor to return image, save elsewhere
def six_cloud_contrast3(img, name, date):
    # Find the mask and black out those pixels.
    mask = Mask.generate_clean_mask()
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


def six_cloud_contrast(img, name, date):
    # Find the mask and black out those pixels.
    mask = Mask.generate_clean_mask()
    img = Mask.apply_mask(mask, img)

    # Closing
    img2 = ndimage.grey_closing(img, size=(2,2))
    #img2 = np.copy(img)

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
    
    imgT = np.copy(img7)
    
    # Experimentally a 240 radius (creating a square slice with the circular
    # image inscribed in the square) IN GENERAL proves to threshold better
    # and with less random nonsense than thresholding the entire square image
    # when clouds are present, and generally worse when they aren't.
    radius = 252
    half = imgT.shape[0] // 2
    topleft = imgT[center[1] - radius:half, center[0] - radius:half]
    bottomleft = imgT[half:center[1] + radius, center[0] - radius:half]
    topright = imgT[center[1] - radius:half, half:center[0] + radius]
    bottomright = imgT[half:center[1] + radius, half:center[0] + radius]
    
    if not np.array_equal(topleft, np.zeros(topleft.shape)):
        thresh = threshold_isodata(topleft)
        topleft2 = np.where(topleft > thresh, 255, 0)
    else:
        topleft2 = np.copy(topleft)
    
    if not np.array_equal(topright, np.zeros(topright.shape)):
        thresh = threshold_isodata(topright)
        topright2 = np.where(topright > thresh, 255, 0)
    else:
        topright2 = np.copy(topright)
    
    if not np.array_equal(bottomleft, np.zeros(bottomleft.shape)):
        thresh = threshold_isodata(bottomleft)
        bottomleft2 = np.where(bottomleft > thresh, 255, 0)
    else:
        bottomleft2 = np.copy(bottomleft)
    
    if not np.array_equal(bottomright, np.zeros(bottomright.shape)):
        thresh = threshold_isodata(bottomright)
        bottomright2 = np.where(bottomright > thresh, 255, 0)
    else:
        bottomright2 = np.copy(bottomright)
    
    imgT = np.zeros(img2.shape)
    imgT[center[1] - radius:half, center[0] - radius:half] = topleft2
    imgT[half:center[1] + radius, center[0] - radius:half] = bottomleft2
    imgT[center[1] - radius:half, half:center[0] + radius] = topright2
    imgT[half:center[1] + radius, half:center[0] + radius] = bottomright2
    
    
    # Threshold the image so we have the clouds white and everything else black.
    #img6 = np.where(img7 > thresh, 255, 0)
    
    img6 = np.copy(imgT)
    
    temp = np.int16(img)
    
    # Row = y, Column = x
    for row in range(0,img6.shape[0]):
        for column in range(0,img4.shape[1]):
        
            x = column - center[0]
            y = center[1] - row
            r = math.sqrt(x**2 + y**2)
            if r <= 240 and img6[row, column] == 255:
                temp[row, column] = 0#temp[row, column] *.5#- 40
            
                if temp[row, column] < 0:
                    temp[row, column] = 0

    img8 = np.uint8(temp)

    loc = 'Images/Cloud/' + str(date)

    ImageIO.save_image(img8, name, loc, 'gray')
    

# Takes in an image as an np ndarray.
# Name and date are for saving purposes
def six_cloud_contrast2(img, name, date):
    
    # Find the mask and black out those pixels.
    mask = Mask.generate_mask()
    img = Mask.apply_mask(mask, img)

    # Closing
    img2 = np.copy(img)#ndimage.grey_opening(img, size=(2,2))

    # Inverts
    img3 = np.subtract(255, img2)

    # Subtract closing from the invert to increase contrast. 
    # If it goes negative I want it to be 0 rather than positive abs of num.
    temp = np.int16(img3) - np.int16(img2)
    img4 = np.where(temp < 0, 0, temp)

    # Subtract the original from the working image.
    # Increases contrast with clouds.
    temp = np.int16(img4) - np.int16(img)
    img5 = np.where(temp < 0, 0, temp)
    
    # Again because if there are no clouds this makes it a bit less conspicuous
    temp = np.int16(img5) - np.int16(img)
    img6 = np.where(temp < 0, 0, temp)
    
    temp = np.copy(img)
    
    # This is kind of cool so I left it here in case someone wants to see.
    #multiple = np.abs(.1 - img6 / 255)

    
    temp = np.where(temp < 0, 0, temp)

    img7 = np.uint8(temp)
    
    
    img62 = ndimage.grey_closing(img6, size=(2,1))
    
    # Generate Figure and Axes objects.
    figure = plt.figure()
    figure.set_size_inches(4, 4)  # 4 inches by 4 inches
    axes = plt.Axes(figure, [0., 0., 1., 1.])  # 0 - 100% size of figure

    # Turn off the actual visual axes for visual niceness.
    # Then add axes to figure
    axes.set_axis_off()
    figure.add_axes(axes)

    # Adds the image into the axes and displays it
    
    
    imgbin = np.where(img62 > 10, 1, 0)
    imgbin = ndimage.binary_opening(imgbin)
    
    imgbin = Mask.apply_mask(mask, imgbin)
    axes.imshow(imgbin, cmap='gray')

    axes.set_aspect('equal')
    
    loc = 'Images/Cloud/' + str(date) + ' - Invert 1'
    ImageIO.save_image(img62, name, loc, 'gray')
    
    loc = 'Images/Cloud/' + str(date) + ' - Binary 1'
    ImageIO.save_image(imgbin, name, loc, 'gray')
    
    struct = [[False, True, False],[True, True, True],[False, True, False]]
    imgbin = ndimage.binary_dilation(imgbin)
    
    # Creates a buffer circle keeping the circle isolated from the background.
    for row in range(0,img.shape[1]):
        for column in range(0,img.shape[0]):
            x = column - center[0]
            y = center[1] - row
            r = math.sqrt(x**2 + y**2)
            if (r < 246) and (r > 241):
                imgbin[row,column] = 0

    loc = 'Images/Cloud/' + str(date) + ' - Binary 2'
    ImageIO.save_image(imgbin, name, loc, 'gray')
    
    # This structure makes it so that diagonally connected pixels are part of
    # the same region.
    struct = [[True, True, True],[True, True, True],[True, True, True]]
    labeled, num_features = ndimage.label(imgbin, structure = struct)
    regionsize = [0] * (np.amax(labeled)+1)
    stars = []
    
    
    for row in range(0,img.shape[1]):
        for column in range(0,img.shape[0]):
            regionsize[labeled[row,column]] += 1
            
            # This finds stars in "cloud" regions
            # Basically, if somewhat bright, and the region is marked "cloud."
            if img[row, column] >= (95) and imgbin[row, column] == 1:
                
                x = column - center[0]
                y = center[1] - row
                r = math.sqrt(x**2 + y**2)
                # Let's save time but not reading extraneous outside the circle
                # stars. Saves time mostly in reducing number of active patches.
                if(r <= 240):
                    circ = Circle((column,row), radius=3, fill=False, edgecolor='green')
                    axes.add_patch(circ)
                    stars.append((row, column))
                    
                
    #stars.append((510,510))
    #stars.append((508,508))

    #circ = Circle(center, radius=241, fill=False, edgecolor='green')
    #axes.add_patch(circ)
    
    file = 'Images/Cloud/20170623 - Circle/' + name
    plt.savefig(file, dpi=128)

    plt.close()
    
    # List length  = # of labeled regions + 1 (# of labeled regions does not
    # include region 0, black sky)
    starnums = [0] * (np.amax(labeled)+1)
    for star in stars:
        regionnum = labeled[star[0], star[1]]
        starnums[regionnum] += 1
        
    # The reason why I use density is mainly because of very small non-clouds.
    # They contain few stars, which rules out a strictly star count method.
    # This, however, is actually density^-1. I.e. it's size/stars rather than
    # stars/size. This is because stars/size is very small sometimes.
    temp = np.copy(labeled)
    #density = np.divide(starnums,regionsize)
    
    # I'm aware of a division by 0 warning here. If a region has no stars, then
    # this divides by 0. In fact this np.where exists to ignore that and set
    # zero star regions to a density of 0, since I ignore those later.
    # Hence I'm supressing the divide by 0 warning for these two lines.
    with np.errstate(divide='ignore'):
        density = np.divide(regionsize, starnums)
        density = np.where(np.asarray(starnums) < 1, 0, density)
    
    # Zeroes out densities < 12
    density = np.where(density < 12, 0, density)
    density[0] = 350
    
    for row in range(0,temp.shape[1]):
        for column in range(0,temp.shape[0]):
            value = labeled[row, column]
            temp[row, column] = density[value]
            
    loc = 'Images/Cloud/' + str(date) + ' - Density'
    ImageIO.save_image(temp, name, loc, 'gray')
    
    # If the value is less than the mean density, we want to mask it in the 
    # "map" image. Hence set it to 0, everything else to 1, and multipy.
    # This keeps the non masks (x*1 = x) and ignores the others (x*0 = 0)
    m = np.mean(density[np.nonzero(density)])
    s = np.std(density[np.nonzero(density)])
    
    
    imgT = np.where(temp < m, 0, 1)
    img6_1 = np.multiply(img6, imgT)
    
    
    loc = 'Images/Cloud/' + str(date) + ' - Corrected'
    ImageIO.save_image(img6_1, name, loc, 'gray')
    
    
    # The thinking here is that the whiter it is in the contrast++ image, the
    # darker it should be in the original. Thus increasing cloud contrast
    # without making it look like sketchy black blobs. 
    multiple = .6 - img6_1 / 255
    temp = np.multiply(img2, multiple)
    
    loc = 'Images/Cloud/' + str(date)
    ImageIO.save_image(temp, name, loc, 'gray')
    
    print(density[np.nonzero(density)])
    print(m)
    print(s)
    print(m + s)

    
date = '20170624'
directory = 'Images/Original/' + date + '/'
files = os.listdir(directory)
file = 'r_ut105647s18240.png'
file = 'r_ut080515s07920.png'
#file = 'r_ut113241s20400.png'
for file in files:
    img = ndimage.imread(directory + file, mode='L')
    six_cloud_contrast2(img, file, date)
    

