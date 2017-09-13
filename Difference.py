from scipy import ndimage
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot


# Reads the two images into ndarrays.
# Importing in RGB mode makes it ignore the A layer.
# If we suddenly have transparency I might have a problem.
img1 = ndimage.imread('conv.png', mode='RGB')
img2 = ndimage.imread('conv2.png', mode='RGB')

# Difference image. Assumed literal mathematical difference for now.
# I encountered a problem previously, in that
# I assumed the type of the array would dynamically change.
# This is python, so that's not wrong per se.
# Anyway turns out it's wrong so I have to cast these to numpy ints.
# I then have to cast back to uints because imshow
# works differently on uint8 and int16.
diffimg = np.uint8(abs(np.int16(img1) - np.int16(img2)))

dpi = 128
y = diffimg.shape[0] / dpi
x = diffimg.shape[1] / dpi

# Generate Figure and Axes objects.
figure = plot.figure()
figure.set_size_inches(x, y)  # 4 inches by 4 inches
axes = plot.Axes(figure, [0., 0., 1., 1.])  # 0 - 100% size of figure


# Turn off the actual visual axes for visual niceness.
# Then add axes to figure
axes.set_axis_off()
figure.add_axes(axes)

# Adds the image into the axes and displays it
axes.imshow(diffimg)


# DPI chosen to have resultant image be the same size as the originals.
# 128*4 = 512
plot.savefig("blah.png", dpi=dpi)

# Show the plot
plot.show()

plot.close()