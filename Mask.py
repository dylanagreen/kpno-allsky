from scipy import ndimage
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import math
import os
from collections import Counter

import ImageIO

# Looks through the images to make the mask and returns a mask array.
def find_mask():
    fileloc = 'Images/Mask/'
    files = os.listdir(fileloc)

    tolerance = 150

    hots = []

    for file in files:
        file = fileloc + file
        img = ndimage.imread(file, mode='L')

        y = 0
        x = 0
        while y < img.shape[1]:
            while x < img.shape[0]:
                # 255 is pure white so accept pixels between
                # white-tolerance and white
                # Y is first value as it's the row value
                if img[y, x] >= (255 - tolerance):
                    hots.append((x, y))

                x += 1
            y += 1
            x = 0

    # Get only the pixels that appear as "hot" in all of the images
    final = []
    for item, num in Counter(hots).items():
        if num == len(files):
            final.append(item)

    return sorted(final)


# Saves a given mask as in image in the Images folder.
def save_mask(mask):
    dest = 'Images/Mask.png'

    newimg = np.zeros((512, 512))
    newimg.fill(100)

    for loc in mask:
        newimg[loc[1], loc[0]] = 255
    
    ImageIO.save_image(newimg, 'Mask.png', 'Images/', 'gray')


# Applies the mask to the image
def apply_mask(mask, img):
    # For masked pixel set the pixel to 0
    for loc in mask:
        img[loc[1], loc[0]] = 0

    return img
