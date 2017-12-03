import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import os
import math
from scipy import ndimage

# Saves an input image with the given name in the folder denoted by location.
# If the image is greyscale, cmap should be 'gray'
def save_image(img, name, location, cmap=None):
    
    if not os.path.exists(location):
        os.makedirs(location)
    
    # DPI chosen to have resultant image be the same size as the originals.
    # 128*4 = 512
    dpi = 128
    y = img.shape[0] / dpi
    x = img.shape[1] / dpi
    
    # Generate Figure and Axes objects.
    figure = plot.figure()
    figure.set_size_inches(x, y)  # 4 inches by 4 inches
    axes = plot.Axes(figure, [0., 0., 1., 1.])  # 0 - 100% size of figure

    # Turn off the actual visual axes for visual niceness.
    # Then add axes to figure
    axes.set_axis_off()
    figure.add_axes(axes)

    # Adds the image into the axes and displays it
    # Then saves
    axes.imshow(img, cmap=cmap)
    
    # If location was passed with / on the end, don't append another one.
    if not location[-1:] == '/':
        name = location + '/' + name
    else:
        name = location + name
    
    # Append .png if it wasn't passed in like that already.
    if not name[-4:] == '.png':
        name = name + '.png'
    
    # Print "saved" after saving, in case saving messes up.
    plot.savefig(name, dpi=dpi)
    print('Saved: ' + name)

    # Show the plot
    #plot.show()

    # Close the plot in case you're running multiple saves.
    plot.close()


# Gets the exposure time of an image.
# 225 is the greyscale value for the yellow in the image.
# The pixel chosen is yellow for .03s and "not yellow" for 6s.
# The second pixel is yellow in .03 and .002
# but due to magic of if blocks that's ok.
def get_exposure(image):
    
    # Handles separate cases for greyscale and RGB images.
    if len(image.shape) == 2:
        pix1 = image[19, 174]
        pix2 = image[17, 119]
    # Greyscale conversion below is the same one used by imread.
    elif len(image.shape) == 3:
        pix1 = image[19, 174]
        pix1 = pix1[0] * 299/1000 + pix1[1] * 587/1000 + pix1[2] * 114/1000
        pix1 = math.floor(pix1)
        
        pix2 = image[17, 119]
        pix2 = pix2[0] * 299/1000 + pix2[1] * 587/1000 + pix2[2] * 114/1000
        pix2 = math.floor(pix2)
    
    if pix1 == 225:
        return '0.3'
    if pix2 == 225:
        return '0.02'
    else:
        return '6'


# Returns the difference image between two images. 
# Black areas are exactly the same in both, white areas are opposite.
# Greyscale/color values are varying levels of difference.
def image_diff(img1, img2):
    # I encountered a problem previously, in that
    # I assumed the type of the array would dynamically change.
    # This is python, so that's not wrong per se.
    # Anyway turns out it's wrong so I have to cast these to numpy ints.
    # I then have to cast back to uints because imshow
    # works differently on uint8 and int16.
    diffimg = np.uint8(abs(np.int16(img1) - np.int16(img2)))

    return diffimg

