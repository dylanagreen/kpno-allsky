import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import os
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
    
    name = location + '/' + name
    
    # Append .png if it wasn't passed in like that already.
    if not name[-4:] == '.png':
        name = name + '.png'
    
    print('Saved: ' + name)
    plot.savefig(name, dpi=dpi)

    # Show the plot
    #plot.show()

    plot.close()
