from scipy import ndimage
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot

import ImageIO

# Reads the two images into ndarrays.
# Importing in RGB mode makes it ignore the A layer.
# If we suddenly have transparency I might have a problem.
img1 = ndimage.imread('Blah01-1.png', mode='RGB')
img2 = ndimage.imread('Blah01-2.png', mode='RGB')

# Difference image. Assumed literal mathematical difference for now.
# I encountered a problem previously, in that
# I assumed the type of the array would dynamically change.
# This is python, so that's not wrong per se.
# Anyway turns out it's wrong so I have to cast these to numpy ints.
# I then have to cast back to uints because imshow
# works differently on uint8 and int16.
diffimg = np.uint8(abs(np.int16(img1) - np.int16(img2)))

ImageIO.save_image(diffimg, 'Difference', 'Images/')