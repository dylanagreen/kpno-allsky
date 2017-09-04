from scipy import ndimage
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import math
import os

import Coordinates
import Mask


center = (256, 252)


# This takes the file and the given date and then transforms it
# from the circle into an eckert-iv projected ra-dec map.
# TODO Refactor this to take in a numpy image and read in the image elsewhere.
def transform(file, date):
    # Read in the file on given date.
    img = ndimage.imread('Images/' + date + '/' + file, mode='L')
    time = Coordinates.timestring_to_obj(date, file)

    # Find the mask and black out those pixels.
    mask = Mask.find_mask()
    img = Mask.apply_mask(mask, img)

    # Read in the ignore image.
    ignore = 'Images/Ignore.png'
    img2 = ndimage.imread(ignore, mode='RGB')

    # This is black background stuff
    rapoints = []
    decpoints = []

    # Just a bunch of ra-dec points for the background.
    ra = 0
    while ra <= 360:
        dec = -90
        while dec <= 90:
            rapoints.append(ra)
            decpoints.append(dec)

            dec += .5
        ra += .5

    x, y = eckertiv(rapoints, decpoints)

    # Sets up the figure and axes objects
    fig = plot.figure(frameon=False)
    fig.set_size_inches(12, 6)

    # We just want the globe to be centered in an image so we turn off the axis.
    ax1 = plot.Axes(fig, [0., 0., 1., 1.])
    ax1.set_axis_off()

    # Scatter for the background
    # (i.e. fills in the rest of the globular shape with black)
    scatter = ax1.scatter(x, y, s=2, color='black')

    # This is the image conversion
    altpoints = []
    azpoints = []
    xpoints = []
    ypoints = []
    colors = []

    for row in range(0, img.shape[0]):
        for column in range(0, img.shape[1]):

            x = column - center[0]
            y = center[1] - row
            r = math.sqrt(x**2 + y**2)

            # Zeros out ignorable objects first
            if(r < 250):
                if np.array_equal(img2[column, row], [244, 66, 235]):
                    img[column, row] = 0
                elif (r > 240):
                    img[column, row] = 0

            if(r < 241):
                # We need to add 0.5 to the r,c coords to get the center
                # of the pixel rather than the top left corner.
                # I've also had to split the xy-radec conversion.
                # The reason for this is that xy-to-altaz doesn't work
                # On np.arrays vectorwise like altaz-to-radec does.
                alt, az = Coordinates.xy_to_altaz(column + 0.5, row + 0.5)
                x, y = Coordinates.camera_conv(column, row, az)

                # x = column
                # y = row
                alt, az = Coordinates.xy_to_altaz(x, y)

                altpoints.append(alt)
                azpoints.append(az)
                xpoints.append(column)
                ypoints.append(row)

    # Convert the alt az to x,y
    rapoints, decpoints = Coordinates.altaz_to_radec(altpoints, azpoints, time)

    # Finds colors for dots.
    for i in range(0, len(rapoints)):
        x = xpoints[i]
        y = ypoints[i]

        colors.append(img[y, x])

    x, y = eckertiv(rapoints, decpoints)

    # Scatter for the image conversion
    scatter = ax1.scatter(x, y, s=1, c=colors, cmap='gray')

    # Date formatting for lower left corner text.
    formatted = date[:4] + '-' + date[4:6] + '-' + date[6:]
    time = file[4:6] + ':' + file[6:8] + ':' + file[8:10]

    # These coord: -265.300085635, -132.582101423
    # are the minimum x and y of the projection.
    # I found them by sorting x and y.
    ax1.text(-290, -143, formatted + '  ut' + time, style='italic')

    # Add the axes to the fig so it gets saved.
    fig.add_axes(ax1)

    # Make sure the folder location exists
    directory = 'Images/Scatter/' + date + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save name.
    conv = directory + file
    plot.savefig(conv, dpi=256)

    print('Saved: ' + conv)

    # Gotta close the plot so we don't memory overflow lol.
    plot.close()


# Newton's method for the mollweide projection.
def mollweide_findtheta(phi, n):
    # First short circuit
    if n == 0:
        return np.arcsin(2*phi/math.pi)

    # Array literally just filled with half pis.
    halfpi = np.empty(len(phi))
    halfpi.fill(math.pi/2)

    theta = mollweide_findtheta(phi, n-1)

    cond1 = np.equal(theta, halfpi)
    cond2 = np.equal(theta, -1*halfpi)
    cond = np.logical_or(cond1, cond2)

    # Choose the original value (pi/2 or neg pi/2) if its true for equality
    # Otherwise use that value's thetanew.
    num = (2 * theta + np.sin(2 * theta) - math.pi*np.sin(phi))
    thetanew = theta - num/(2 + 2 * np.cos(2 * theta))
    thetanew = np.where(cond, phi, thetanew)

    return thetanew


# Dec = lat = phi
# Ra = long = lambda
# Should work on lists/np arrays
# This is the mollweide transformation.
def mollweide(ra, dec):
    # Center latitude
    center = math.radians(180)

    theta = mollweide_findtheta(np.radians(dec), 2)

    R = 100

    # Mollweide conversion functions.
    # a is the minor axis of an ellipse, hence the variable.
    a = R * math.sqrt(2)
    x = (2 * a / math.pi)*(np.subtract(np.radians(ra), center))*np.cos(theta)
    y = a * np.sin(theta)

    return(x, y)


# Newton's method for eckert-iv proection.
def eckertiv_findtheta(phi, n):
    # First short circuit
    if n == 0:
        return phi/2

    pi = math.pi

    # Array literally just filled with half pis.
    halfpi = np.empty(len(phi))
    halfpi.fill(pi/2)

    theta = eckertiv_findtheta(phi, n-1)

    # Condition for the angle is pi/2 OR -pi/2
    cond1 = np.equal(theta, halfpi)
    cond2 = np.equal(theta, -1*halfpi)
    cond = np.logical_or(cond1, cond2)

    # Choose the original value (pi/2 or -pi/2) if its true for equality
    # Otherwise use that value's thetanew.
    # This is the eckertiv theta finding Newton's method.
    # It's been broken up for style.
    s_theta = np.sin(theta)
    c_theta = np.cos(theta)
    num = theta + np.multiply(s_theta, c_theta) + 2 * s_theta - (2 + pi/2) * np.sin(phi)
    denom = 2 * c_theta * (1 + c_theta)
    thetanew = theta - num/denom
    thetanew = np.where(cond, phi, thetanew)

    return thetanew


def eckertiv(ra, dec):
    # Center latitude
    center = math.radians(180)

    # n = 5 seems to be sufficient for the shape.
    # This doesn't converge as quickly as Mollweide
    theta = eckertiv_findtheta(np.radians(dec), 5)

    R = 100

    # For readability sake
    coeff = 1/math.sqrt(math.pi*(4+math.pi))

    # Eckert IV conversion functions.
    x = 2 * R * coeff * np.subtract(np.radians(ra), center) * (1 + np.cos(theta))
    y = 2 * R * math.pi * coeff * np.sin(theta)

    return(x, y)

date = '20170829'
directory = 'Images/' + date + '/'
files = os.listdir(directory)

transform('r_ut033749s07680.png', date)

images = []
# Loop for transforming a whole day.
#for file in files:
    #transform(file)
