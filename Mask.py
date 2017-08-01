from scipy import ndimage
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import math
import os
from collections import Counter

fileloc = 'Images/Temp/'
files = os.listdir(fileloc)

tolerance = 5

hots = []

for file in files:
    file = fileloc + file
    img = ndimage.imread(file, mode = 'L')

    y = 0
    x = 0
    while y < img.shape[1]:
        while x < img.shape[0]:
            
            #print(img[x,y])
            # 255 is pure white so accept pixels between white-tolerance and white
            # Y is first value as its the row value
            if img[y,x] >= (255 - tolerance):
                hots.append((x,y))
        
            x += 1
        y += 1
        x = 0

# Get only the pixels that appear as "hot" in all of the images
final = []
for item,num in Counter(hots).items():
    if num == len(files):
        final.append(item)
        
print(sorted(final))
print(len(final))

dest = 'Images/Mask.png'

newimg = np.zeros((512,512))
newimg.fill(100)

for loc in final:
    newimg[loc[1],loc[0]] = 255

figure = plot.figure()
figure.set_size_inches(4,4) # 4 inches by 4 inches
axes = plot.Axes(figure,[0.,0.,1.,1.]) # 0 - 100% size of figure

# Turn off the actual visual axes for visual niceness.
# Then add axes to figure
axes.set_axis_off()
figure.add_axes(axes)

# Adds the image into the axes and displays it
axes.imshow(newimg, cmap = 'gray')

plot.savefig(dest, dpi = 128)