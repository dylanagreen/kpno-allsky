import math
from scipy import ndimage
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import math
import os

import Coordinates
import Mask

import imageio

center = (256, 252)
date = '20170829'

def transform(file):
    img = ndimage.imread('Images/' + date + '/' + file, mode = 'L')
    
    
    time = Coordinates.timestring_to_obj(date,file)
    
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

    # Finds colors for dots.
    for i in range(0,len(rapoints)):
        x = xpoints[i]
        y = ypoints[i]
    
        colors.append(img[y,x])
    
    x,y = eckertiv(rapoints,decpoints)
    # Scatter Plot Style
    fig, ax1 = plot.subplots()


    conv = 'Images/Scatter/' + date + '/' + file[:-4] + '.png'

    # Sets up the scatter plot
    scatter = ax1.scatter(x,y, s = 2, c = colors, cmap = 'gray')
    ax1.set_ylabel("Dec")
    ax1.set_xlabel("Ra")

    # Celestial Horizon
    ax1.set_yticks([0], minor=True)
    #ax1.grid(True, axis = 'y', which = 'minor')

    # Tick marks
    #ax1.set_xticks(range(0,370,20))

    #ax1.set_yticks(range(-90,100,20))

    # Adjust for nice white space.
    #fig.subplots_adjust(left = 0.06, right = .98, top = .96, bottom = .1)
    
    #fig.subplots_adjust(left = 0.09, right = .9, top = .92, bottom = .13)

    # Background to black and title
    ax1.set_facecolor('black')
    ax1.set_title(date + '-' + file[:-4])

    fig.set_size_inches(12,6)
    
    # Make sure the folder location exists
    dir = 'Images/Scatter/' + date + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    plot.savefig(conv, dpi = 256)

    print('Saved: ' + conv)
    #plot.show()
    
    # Gotta close the plot in case you're looping so we don't memory overflow lol.
    plot.close()
    
    
def points():
    
    rapoints = []
    decpoints = []
    
    i = 0
    while i <= 360:
        j = -90
        while j <= 90:
            rapoints.append(i)
            decpoints.append(j)
            
            j += 1
        i += 1
    
    
    x,y = eckertiv(rapoints,decpoints)
    # Scatter Plot Style
    fig, ax1 = plot.subplots()


    date = '1'
    conv = 'Images/Scatter/' + date + '/EckertIV-test.png'

    # Sets up the scatter plot
    scatter = ax1.scatter(x,y, s = 2)#, c = colors, cmap = 'gray')
    ax1.set_ylabel("y")
    ax1.set_xlabel("x")

    # Celestial Horizon
    #ax1.set_yticks([-100*math.sqrt(2),100*math.sqrt(2)], minor=True)
    #ax1.grid(True, axis = 'y', which = 'minor')
    
    #ax1.set_xticks([-200*math.sqrt(2),200*math.sqrt(2)], minor=True)
    #ax1.grid(True, axis = 'x', which = 'minor')

    # Tick marks
    #ax1.set_xticks(range(0,370,20))
    #ax1.set_yticks(range(-90,100,20))


    # Background to black and title
    ax1.set_facecolor('black')
    #ax1.set_title(date + '-' + file[:-4])

    fig.set_size_inches(12,6)
    
    # Make sure the folder location exists
    dir = 'Images/Scatter/' + date + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    plot.savefig(conv, dpi = 256)

    print('Saved: ' + conv)
    #plot.show()
    
    # Gotta close the plot in case you're looping so we don't memory overflow lol.
    plot.close()
    

# Find theta method for the mollweide projection.
def mollweide_findtheta(phi,n): 
    
    # First short circuit   
    if n == 0:
        return np.arcsin(2*phi/math.pi)

    # Array literally just filled with half pis.
    halfpi = np.empty(len(phi))
    halfpi.fill(math.pi/2)
    
    
    theta = mollweide_findtheta(phi, n-1)
    
    cond1 = np.equal(theta,halfpi)
    cond2 = np.equal(theta, -1*halfpi)
    cond = np.logical_or(cond1,cond2)

    # Choose the original value (pi/2 or neg pi/2) if its true for equality
    # Otherwise run that thetanew boys.
    thetanew = np.where(cond, phi, theta - (2*theta + np.sin(2*theta) - math.pi*np.sin(phi))/(2+2*np.cos(2*theta)))
    
    return thetanew


# Dec = lat = phi
# Ra = long = lambda
# Should work on lists/np arrays
# This is the mollweide transformation.
def mollweide(ra,dec):
    lambda_0 = math.radians(180)
    
    theta = mollweide_findtheta(np.radians(dec),2)
    
    R = 100
    
    x = (R*2*math.sqrt(2)/math.pi)*(np.subtract(np.radians(ra),lambda_0))*np.cos(theta)
    y = R*math.sqrt(2)*np.sin(theta)
    
    x1 = sorted(x)
    y1 = sorted(y)
    
    print(x1[0])
    print(x1[-1])
    print(y1[0])
    print(y1[-1])
    
    print(100*math.sqrt(2))
    
    return(x,y)
    

def eckertiv_findtheta(phi,n):
    
    # First short circuit   
    if n == 0:
        return phi/2

    pi = math.pi
    
    # Array literally just filled with half pis.
    halfpi = np.empty(len(phi))
    halfpi.fill(pi/2)
    
    
    theta = eckertiv_findtheta(phi, n-1)
    
    cond1 = np.equal(theta,halfpi)
    cond2 = np.equal(theta, -1*halfpi)
    cond = np.logical_or(cond1,cond2)

    # Choose the original value (pi/2 or neg pi/2) if its true for equality
    # Otherwise run that thetanew boys.
    thetanew = theta - (theta + np.multiply(np.sin(theta), np.cos(theta)) + 2 * np.sin(theta)- (2 + pi/2)*np.sin(phi))/(2*np.cos(theta)*(1+np.cos(theta)))
    thetanew = np.where(cond, phi, thetanew)
    
    return thetanew
    
    

def eckertiv(ra,dec):
    lambda_0 = math.radians(180)
    
    # n = 5 seems to be sufficient for the shape.
    # This doesn't converge as quickly as Mollweide
    theta = eckertiv_findtheta(np.radians(dec),5)
    
    R = 100
    
    coeff = 1/math.sqrt(math.pi*(4+math.pi))
    
    x = 2 * R * coeff * np.subtract(np.radians(ra),lambda_0)*(1 + np.cos(theta))
    y = 2 * R * math.pi * coeff* np.sin(theta)
    
    return(x,y)

directory = 'Images/Scatter/' + date + '/'
#files = os.listdir(directory)

transform('r_ut092452s28560.png')

#points()


images = []
#for file in files:
    #func(file)
    #images.append(imageio.imread(directory + file))
    
#imageio.mimsave('blah.gif', images, duration = 1/27)

