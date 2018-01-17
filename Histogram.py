from scipy import ndimage
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
import math
import os
import Mask

from matplotlib.patches import Circle

center = (256, 252)

def histogram(file):
    
    date = '20171108'
    img = ndimage.imread('Images/Original/' + date + '/' + file, mode = 'L')
    mask = Mask.generate_full_mask()
    mask = 1 - mask
    
    # Converts the 1/0 array to True/False so it can be used as an index.
    mask = np.ma.make_mask(mask)
    #img1 = Mask.apply_mask(mask, img1)
    
    #hist = np.histogram(img1, bins = 255)
    img1 = img[mask]

    #print(sum(hist[0]))
    fig, ax = plt.subplots(1, 2)
    
    fig.set_size_inches(10,5)

    # Turn off the actual visual axes for visual niceness.
    # Then add axes to figure
    ax[0].set_axis_off()
    ax[0].imshow(img, cmap='gray')
    #figure.add_axes(axes)
    
    
    bins = list(range(0,256))
    ax[1].hist(img1.flatten(), bins = bins, color = 'blue', log = True)
    #plt.show()
    
    name = 'Images/Histogram/' + date + '/' + file
    
    if not os.path.exists('Images/Histogram/' + date + '/'):
        os.makedirs('Images/Histogram/' + date + '/')
    
    plt.savefig(name, dpi=256)
    
    print('Saved: ' + name)
    
    plt.close()

def test(file):
    
    img1 = ndimage.imread('Images/Original/20171108/' + file, mode = 'L')
    mask = Mask.generate_mask()
    img1 = Mask.apply_mask(mask, img1)
    img2 = img1.flatten()
    
    blah = [False] * 256
    blah2 = {}
    
    for i in range(0, 256):
        print(i)
        if i in img2:
            #blah[i] = True
            blah2[i] = True
        else:
            #blah[i] = False
            blah2[i] = False
    
    #print(blah)
    print(blah2)


directory = 'Images/Original/20171108/'
files = os.listdir(directory)

for file in files:
    histogram(file)
    
#test('r_ut012616s16560.png')
