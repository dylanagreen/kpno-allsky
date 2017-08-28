import math
from scipy import ndimage
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import math
import os

import Coordinates
import Mask

center = (256, 252)

file = 'Images/Map.png'

img = ndimage.imread(file, mode = 'L')

newimg = np.zeros((361,721))

date = '20170822'
filename = 'r_ut044514s10080.png'
time = Coordinates.timestring_to_obj(date,filename)


mask = Mask.findmask()
img = Mask.applymask(mask,img)


ignore = 'Images/Ignore.png'
img2 = ndimage.imread(ignore, mode = 'RGB')
for row in range(0,img2.shape[0]):
    for column in range(0,img2.shape[1]):
        
        x = column - center[0]
        y = center[1] - row
        r = math.sqrt(x ** 2 + y ** 2)
        
        if(r < 241):
            if (np.array_equal(img2[column,row],[244, 66, 235])):
                img[column,row] = 0


altpoints = []
azpoints = []
xpoints = []
ypoints = []
colors = []

for row in range(0,img.shape[0]):
    for column in range(0,img.shape[1]):
        
        x = column - center[0]
        y = center[1] - row
        r = math.sqrt(x ** 2 + y ** 2)
        
        
        if(r < 241):
            
            # We need to add 0.5 to the r,c coords to get the center of the pixel rather than the top left corner.
            alt,az = Coordinates.xy_to_altaz(column + 0.5, row  + 0.5)
            x,y = Coordinates.camera_conv(column,row,az)
    
            #x = column
            #y = row
            alt,az = Coordinates.xy_to_altaz(x,y)
            
            #print(az)
            
            altpoints.append(alt)
            azpoints.append(az)
            xpoints.append(column)
            ypoints.append(row)
            
    #print(row)


rapoints,decpoints = Coordinates.altaz_to_radec(altpoints,azpoints,time)




for i in range(0,len(rapoints)):
    ra = np.int32(round(2*rapoints[i]))
    dec = np.int32(round(2*(90-decpoints[i])))
    x = xpoints[i]
    y = ypoints[i]
    
    colors.append(img[y,x])
    newimg[dec,ra] = img[y,x]


# Scatter Plot Style
fig, ax1 = plot.subplots()


conv = date + '-' + filename[:-4] + '-scatter.png'

# Sets up the scatter plot
scatter = ax1.scatter(rapoints,decpoints, s = 2, c = colors, cmap = 'gray')
ax1.set_ylabel("Dec")
ax1.set_xlabel("Ra")

# Celestial Horizon
ax1.set_yticks([0], minor=True)
ax1.grid(True, axis = 'y', which = 'minor')

# Tick marks
ax1.set_xticks(range(0,370,20))

ax1.set_yticks(range(-90,100,20))

# Adjust for nice white space.
fig.subplots_adjust(left = 0.06, right = .98, top = .96, bottom = .1)

# Background to black and title
ax1.set_facecolor('black')
ax1.set_title(date + '-' + filename[:-4])

fig.set_size_inches(12,6)
plot.savefig(conv, dpi = 256)


#plot.show()


# Image Style
dpi = 128
y = newimg.shape[0] / dpi
x = newimg.shape[1] / dpi

# Generate Figure and Axes objects.
figure = plot.figure()
figure.set_size_inches(x,y) # 4 inches by 4 inches
axes = plot.Axes(figure,[0.,0.,1.,1.]) # 0 - 100% size of figure


# Turn off the actual visual axes for visual niceness.
# Then add axes to figure
axes.set_axis_off()
figure.add_axes(axes)

# Adds the image into the axes and displays it
axes.imshow(newimg, cmap = 'gray')

conv = date + '-' + filename[:-4] + '-img.png'

#print(conv)
# DPI chosen to have resultant image be the same size as the originals.
plot.savefig(conv, dpi = dpi)

#plot.show()