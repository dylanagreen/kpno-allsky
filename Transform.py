import math
import os
import ast
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
from scipy import ndimage
from matplotlib.patches import Polygon

import Coordinates
import Mask
import Clouds


center = (256, 252)


# This takes the file and the given date and then transforms it
# from the circle into an eckert-iv projected ra-dec map.
# TODO Refactor this to take in a numpy image and read in the image elsewhere.
def transform(file, date):
    # Read in the file on given date.
    img = ndimage.imread('Images/Original/' + date + '/' + file, mode='L')
    time = Coordinates.timestring_to_obj(date, file)

    #img = Clouds.cloud_contrast(img)
    
    
    # Find the mask and black out those pixels.
    # Contrasting the clouds already masks.
    #mask = Mask.generate_mask()
    #img = Mask.apply_mask(mask, img)

    # Sets up the figure and axes objects
    fig = plot.figure(frameon=False)
    fig.set_size_inches(12, 6)

    # We just want the globe to be centered in an image so we turn off the axis.
    ax1 = plot.Axes(fig, [0., 0., 1., 1.])
    ax1.set_axis_off()

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

    # Scatter for the background
    # (i.e. fills in the rest of the globular shape with black)
    x, y = eckertiv(rapoints, decpoints)
    scatter = ax1.scatter(x, y, s=2, color='black')

    # This is the image conversion
    altpoints = []
    azpoints = []
    xpoints = []
    ypoints = []

    for row in range(0, img.shape[0]):
        for column in range(0, img.shape[1]):

            x = column - center[0]
            y = center[1] - row
            r = math.sqrt(x**2 + y**2)

            # Zeros out ignorable objects first
            if(r > 241):
                img[row, column] = 0
            # Only want points in the circle to convert
            else:
                xpoints.append(column)
                ypoints.append(row)

    # We need to add 0.5 to the r,c coords to get the center of the pixel
    # rather than the top left corner.
    # Convert the alt az to x,y
    x = np.add(np.asarray(xpoints), 0.5)
    y = np.add(np.asarray(ypoints), 0.5)
    rapoints, decpoints = Coordinates.xy_to_radec(x, y, time)

    # Finds colors for dots.
    colors = []
    for i in range(0, len(rapoints)):

        # This block changes the ra so that the projection is centered at
        # ra = 360-rot.
        # The reason for this is so the outline survey area is 2 rather than 3
        # polygons.
        rot = 60
        if rapoints[i] > (360-rot):
            rapoints[i] = rapoints[i] + rot - 360
        else:
            rapoints[i] = rapoints[i] + rot

        x = xpoints[i]
        y = ypoints[i]

        colors.append(img[y, x])

    # Scatter for the image conversion
    x, y = eckertiv(rapoints, decpoints)
    scatter = ax1.scatter(x, y, s=1, c=colors, cmap='gray')

    # Add the contours
    ax1 = contours(ax1, time)

    # Date formatting for lower left corner text.
    formatted = date[:4] + '-' + date[4:6] + '-' + date[6:]
    time = file[4:6] + ':' + file[6:8] + ':' + file[8:10]

    # These coord: -265.300085635, -132.582101423
    # are the minimum x and y of the projection.
    # I found them by sorting x and y.
    ax1.text(-290, -143, formatted + '  ut' + time, style='italic')

    patches = hull_patch()
    for patch in patches:
        ax1.add_patch(patch)

    # Add the axes to the fig so it gets saved.
    fig.add_axes(ax1)

    # Make sure the folder location exists
    directory = 'Images/Transform/' + date + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save name.
    conv = directory + file

    # Want it to be 1920 wide.
    dpi = 1920 / (fig.get_size_inches()[0])
    plot.savefig(conv, dpi=dpi)

    print('Saved: ' + conv)

    # Gotta close the plot so we don't memory overflow lol.
    plot.close()


# Adds 0-30-60 degree alt contours to the axis passed in.
# Time parameter is required for altaz -> radec conversion.
def contours(axis, time):

    # Loop runs over all the alts.
    # Resets the arrays at the start, creates alt/az for that alt value.
    for alt in range(0,90,30):
        # We need it to not connect the different contours so they have to be
        # added seperately.
        altpoints = []
        azpoints = []
        for az in range(0,360,1):
            altpoints.append(alt)
            azpoints.append(az)

        rapoints, decpoints = Coordinates.altaz_to_radec(altpoints, azpoints, time)

        # Rotation block
        # Centers contours at 60 degrees ra.
        for i in range(0, len(rapoints)):
            rot = 60
            if rapoints[i] > (360 - rot):
                rapoints[i] = rapoints[i] + rot - 360
            else:
                rapoints[i] = rapoints[i] + rot

        # Don't sort the 60 contour since it's a complete circle.
        if not alt == 60:
            # Sorting by ra so that the left and right edges don't connect.
            points = []
            for i in range(0,len(rapoints)):
                points.append((rapoints[i], decpoints[i]))

            points = sorted(points)

            # Condensing lines using magic python list comprehension.
            rapoints = [point[0] for point in points]
            decpoints = [point[1] for point in points]

            x, y = eckertiv(rapoints, decpoints)

            # 42f44e is super bright green.
            scatter = axis.plot(x,y,c='#42f44e')

        # The 60 contour needs to be two plots if it gets seperated by the edge.
        else:
            temp = sorted(rapoints)
            # Basically if the difference in the least and most is almost the
            # entire image then seperate
            # "Lower" = leftside, "Upper" = rightside
            if temp[-1] - temp[0] > 350:
                lowerra = []
                lowerdec = []
                upperra = []
                upperdec = []
                for i in range(0,len(rapoints)):
                    if rapoints[i] < 180:
                        lowerra.append(rapoints[i])
                        lowerdec.append(decpoints[i])
                    else:
                        upperra.append(rapoints[i])
                        upperdec.append(decpoints[i])

                # Clockwise sorting is necessary here to prevent the top and
                # Bottom ends on either edge from joining.
                # Left needs to be sorted from negative x.
                lowerra, lowerdec = clockwise_sort(lowerra, lowerdec)
                x, y = eckertiv(lowerra, lowerdec)
                scatter = axis.plot(x,y,c='#42f44e')

                # Right needs to be sorted from the positive x.
                upperra, upperdec = clockwise_sort(upperra, upperdec, True)
                x, y = eckertiv(upperra, upperdec)
                scatter = axis.plot(x,y,c='#42f44e')

            else:
                x, y = eckertiv(rapoints, decpoints)
                scatter = axis.plot(x,y,c='#42f44e')

    return axis


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


# Returns a matplotlib patch for each of the two DESI view polygons.
def hull_patch():
    f = open('hull.txt', 'r')
    
    # Converts the string representation of the list to a list of points.
    left = f.readline()
    left = ast.literal_eval(left)
    right = f.readline()
    right = ast.literal_eval(right)

    # Zorder parameter ensures the patches are on top of everything.
    patch1 = Polygon(left, closed = True, fill = False,
                     edgecolor='red',lw=2, zorder=4)
    patch2 = Polygon(right, closed = True, fill = False,
                     edgecolor='red',lw=2, zorder=4)

    f.close()

    return [patch1, patch2]


# This function sorts a set of values clockwise from the center.
# Pos defines whether or not the sort sorts anticlockwise from the positive x
# Or clockwise from the negative x.
# Anticlockwise = True, clockwise = False
def clockwise_sort(ra, dec, positive = False):
    x = sorted(ra)
    y = sorted(dec)

    # Finds the center of the circle ish object
    centerx = (x[0] + x[-1])/2
    centery = (y[0] + y[-1])/2

    x = np.subtract(ra, centerx)
    y = np.subtract(dec, centery)

    # Creates polar nonsense
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Reshape to stack
    r = np.reshape(r, (len(r), 1))
    theta = np.reshape(theta, (len(theta), 1))

    # If we want to sort from pos x, we need to ensure that the negative angles
    # Are actually big positive angles.
    if positive:
        cond = np.less(theta, 0)
        theta = np.where(cond, theta + 2 * np.pi, theta)

    # Stack into list of form (theta,r) (we want to sort theta first)
    stack = np.hstack((theta,r))

    stack2 = []
    for i in stack:
        stack2.append(tuple(i))

    # Standard python sort by theta
    stack2 = sorted(stack2)

    # Now we just have to convert back!
    # Slice out theta and r
    stack2 = np.array(stack2)
    theta = stack2[:,0]
    r = stack2[:,1]

    x = r * np.cos(theta) + centerx
    y = r * np.sin(theta) + centery
    return (x,y)

if __name__ == "__main__":
    date = '20171108'
    directory = 'Images/Original/' + date + '/'
    files = os.listdir(directory)
    
    # Loop for transforming a whole day.
    for file in files:
        transform(file, date)
