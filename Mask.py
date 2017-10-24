from scipy import ndimage
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import math
import os
from collections import Counter

import ImageIO

center = (256, 252)

# Looks through the median images to make the mask and returns a mask array.
# "Clean" in this case means no horizon objects.
def generate_clean_mask():
    fileloc = 'Images/Mask/'
    files = os.listdir(fileloc)

    tolerance = 160
    mask = []

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
                    mask.append((x, y))

                x += 1
            y += 1
            x = 0

    # Get only the pixels that appear as "hot" in all of the images
    # Essentially means we only want the duplicates in the list that are
    # duplicated the same amount of times as the number of files. 
    final = []
    for item, num in Counter(mask).items():
        if num == len(files):
            final.append(item)

    return final
    
# Gets the "clean" mask and then adds the horizon objects to it.
def generate_mask():
    mask = generate_clean_mask()
    
    # Read in the ignore image.
    ignore = ndimage.imread('Images/Ignore.png', mode='RGB')
    
    y = 0
    x = 0
    while y < ignore.shape[1]:
        while x < ignore.shape[0]:
            
            x1 = x - center[0]
            y1 = center[1] - y
            r = math.sqrt(x1**2 + y1**2)
            
            # Ignore horizon objects (which have been painted pink)
            # Only want the horizon objects actually in the circle.
            # Avoids unnecessary pixels.
            if np.array_equal(ignore[y, x], [244, 66, 235]) and r < 242:
                mask.append((x, y))
            x += 1
        y += 1
        x = 0
    
    return mask


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
    img = np.copy(img)
    # For masked pixel set the pixel to 0
    for loc in mask:
        img[loc[1], loc[0]] = 0

    return img
