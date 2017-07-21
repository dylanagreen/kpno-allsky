from scipy import ndimage
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import math

def transform(x,y):
    center = (256, 252)

    # Point adjusted based on the center being at... well... the center. And not the top left. In case you were confused.
    # Y is measured from the top stop messing it up.
    pointadjust = (x - center[0], center[1] - y)
    
    r = math.hypot(pointadjust[0],pointadjust[1])

    az = math.atan2((pointadjust[0]),(pointadjust[1]))

    rpoints = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 11.6]
    thetapoints = [0, 3.58, 7.17, 10.76, 14.36, 17.98, 21.62, 25.27, 28.95, 32.66, 36.40, 40.17, 43.98, 47.83, 51.73, 55.67, 59.67, 63.72, 67.84, 72.03, 76.31, 80.69, 85.21, 89.97,90]

    rtemp = r * 11.6 / 239 # pixel to mm conversion
    
    # This interpolates the value from the two on either side of it.
    theta = np.interp(rtemp, xp = rpoints, fp = thetapoints)
    
    

    # atan2 ranges from -pi to pi so we have to shift the -pi to 0 parts up 2 pi.
    if pointadjust[0] < 0:
        az += 2 * math.pi
    
    if theta == 0:
        rnew = 0
    else:
        rnew = 2 * math.tan(theta/2) * r / theta
    
    rnew = rnew * 239/8.60280601199962
    
    xnew = round(-rnew * math.sin(az))
    ynew = round(rnew * math.cos(az))
    
    
    return (xnew + center[0], center[1] - ynew)

fileloc = 'Images/' + '20170721' + '/' + 'r_ut043526s01920' + '.png'
dest = 'Images/Transform.png'

img = ndimage.imread(fileloc, mode = 'RGB')
newimg = np.zeros((512,512,3))
newimg.fill(100)

y = 0
x = 0
for row in img:
    for column in img:
        
        xnew, ynew = transform(x,y)
        
        # Make sure they're in the image.....
        if abs(ynew) < 512 and abs(xnew) < 512:
            newimg[ynew,xnew] = img[y,x]
        
        x += 1
    y += 1
    x = 0

figure = plot.figure()
figure.set_size_inches(4,4) # 4 inches by 4 inches
axes = plot.Axes(figure,[0.,0.,1.,1.]) # 0 - 100% size of figure

# Turn off the actual visual axes for visual niceness.
# Then add axes to figure
axes.set_axis_off()
figure.add_axes(axes)

# Adds the image into the axes and displays it
axes.imshow(newimg)

plot.savefig(dest, dpi = 128)

# Show the plot
#plot.show()

