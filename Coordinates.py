import math
import os
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.time.core as aptime
from astropy import units as u
import numpy as np

from matplotlib.patches import Circle, Rectangle
import matplotlib.image as image
import matplotlib.pyplot as plot
from scipy import ndimage

import Mask

# Globals

# Center of the circle found using super accurate photoshop layering technique
center = (256, 252)

# r - theta table.
rpoints = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,
           5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10,
           10.5, 11, 11.5, 11.6]
thetapoints = [0, 3.58, 7.17, 10.76, 14.36, 17.98, 21.62, 25.27,
               28.95, 32.66, 36.40, 40.17, 43.98, 47.83, 51.73,
               55.67, 59.67, 63.72, 67.84, 72.03, 76.31, 80.69,
               85.21, 89.97, 90]


# Returns a tuple of form (alt, az)
def xy_to_altaz(x, y):
    # Converts lists/numbers to np ndarrays for vectorwise math.
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Point adjusted based on the center being at... well... the center.
    # And not the top left. In case you were confused.
    # Y is measured from the top stop messing it up.
    pointadjust = (x - center[0], center[1] - y)

    # We use -x here because the E and W portions of the image are flipped
    az = np.arctan2(-pointadjust[0], pointadjust[1])

    # atan2 ranges from -pi to pi but we need 0 to 2 pi.
    # So values with alt < 0 need to actually be the range from pi to 2 pi
    cond = np.less(pointadjust[0], 0)
    az = np.where(cond, az + 2 * np.pi, az)
    az = np.degrees(az)

    # Pythagorean thereom boys.
    r = np.sqrt(pointadjust[0]**2 + pointadjust[1]**2)

    # 90- turns the angle from measured from the vertical
    # to measured from the horizontal.
    # This interpolates the value from the two on either side of it.
    r = r * 11.6 / 240  # Magic pixel to mm conversion rate
    alt = 90 - np.interp(r, xp=rpoints, fp=thetapoints)
    
    # For now if r is on the edge of the circle or beyond
    # we'll have it just be 0 degrees. (Up from horizontal)
    cond = np.greater(r, 240)
    alt = np.where(cond, 0, alt)

    # Az correction
    az = az + .94444
    
    return (alt.tolist(), az.tolist())


# Returns a tuple of form (ra, dec)
def altaz_to_radec(alt, az, time):
    assert type(time) is aptime.Time, "Time should be an astropy Time Object."

    # This is the latitude/longitude of the camera
    camera = (31.959417 * u.deg, -111.598583 * u.deg)

    cameraearth = EarthLocation(lat=camera[0], lon=camera[1],
                                height=2120 * u.meter)

    alt = alt * u.deg
    az = az * u.deg
    altazcoord = SkyCoord(alt=alt, az=az, frame='altaz',
                          obstime=time, location=cameraearth)
    radeccoord = altazcoord.icrs

    return (radeccoord.ra.degree, radeccoord.dec.degree)


# Returns a tuple of form (alt, az)
def radec_to_altaz(ra, dec, time):
    assert type(time) is aptime.Time, "Time should be an astropy Time Object."

    # This is the latitude/longitude of the camera
    camera = (31.959417 * u.deg, -111.598583 * u.deg)

    cameraearth = EarthLocation(lat=camera[0], lon=camera[1],
                                height=2120 * u.meter)

    # Creates the SkyCoord object
    radeccoord = SkyCoord(ra=ra, dec=dec, unit='deg', obstime=time,
                          location=cameraearth, frame='icrs',
                          temperature=5 * u.deg_C, pressure=78318 * u.Pa)

    # Transforms
    altazcoord = radeccoord.transform_to('altaz')

    return (altazcoord.alt.degree, altazcoord.az.degree)


# Returns a tuple of form (x,y)
def altaz_to_xy(alt, az):
    alt = np.asarray(alt)
    az = np.asarray(az)
    
    # Approximate correction (due to distortion of lens?)
    az = az - .94444

    # Reverse of r interpolation
    r = np.interp(90 - alt, xp=thetapoints, fp=rpoints)

    r = r * 240 / 11.6  # mm to pixel rate

    # Angle measured from vertical so sin and cos are swapped from usual polar.
    # These are x,ys with respect to a zero.
    x = -1 * r * np.sin(np.radians(az))
    y = r * np.cos(np.radians(az))

    # y is measured from the top!
    x = x + center[0]
    y = center[1] - y
    pointadjust = (x.tolist(), y.tolist())

    return pointadjust


# Returns a tuple of form (x,y)
# Time must be an astropy Time object.
def radec_to_xy(ra, dec, time):
    alt, az = radec_to_altaz(ra, dec, time)
    x, y = altaz_to_xy(alt, az)
    return galactic_conv(x, y, az)


# Returns a tuple of form (ra,dec)
# Time must be an astropy Time object.
def xy_to_radec(x, y, time):
    alt, az = xy_to_altaz(x, y)
    x, y = camera_conv(x, y, az)
    alt, az = xy_to_altaz(x, y)
    
    return altaz_to_radec(alt, az, time)


# Converts a file name to a time object.
def timestring_to_obj(date, filename):
    # Add the dashes
    formatted = date[:4] + '-' + date[4:6] + '-' + date[6:]

    # Extracts the time from the file name.
    # File names seem to be machine generated so this should not break.
    # Hopefully.
    time = filename[4:6] + ':' + filename[6:8] + ':' + filename[8:10]

    formatted = formatted + ' ' + time

    return Time(formatted)


# Converts from galactic r, expected az to camera r, actual az
def galactic_conv(x, y, az):
    y = np.asarray(y)
    x = np.asarray(x)
    az = np.asarray(az)
    
    # Convert to center relative coords.
    x = x - center[0]
    y = center[1] - y

    r = np.sqrt(x**2 + y**2)
    az = az - .94444

    # This was the best model I came up with.
    r = r + 2.369 * np.cos(np.radians(0.997 * (az - 42.088))) + 0.699
    az = az + 0.716 * np.cos(np.radians(1.015 * (az + 31.358))) - 0.181

    x = -1 * r * np.sin(np.radians(az))
    y = r * np.cos(np.radians(az))

    # Convert to top left relative coords.
    x = x + center[0]
    y = center[1] - y

    return (x.tolist(), y.tolist())


def camera_conv1(x, y, az):
    y = np.asarray(x)
    x = np.asarray(y)
    return (x.tolist(), y.tolist())

# Converts from camera r,az to galactic r,az
def camera_conv(x, y, az):
    y = np.asarray(y)
    x = np.asarray(x)
    az = np.asarray(az)
    
    # Convert to center relative coords.
    x = x - center[0]
    y = center[1] - y

    r = np.sqrt(x**2 + y**2)

    # You might think that this should be + but actually no.
    # Mostly due to math down below in the model this works better as -.
    az = az - .94444  # - 0.286375

    az = np.subtract(az, 0.731 * np.cos(np.radians(0.993 * (az + 34.5)))) + 0.181
    r = np.subtract(r, 2.358 * np.cos(np.radians(0.99 * (az - 40.8)))) - 0.729

    x = -1 * r * np.sin(np.radians(az))
    y = r * np.cos(np.radians(az))

    # Convert to top left relative coords.
    x = x + center[0]
    y = center[1] - y

    return (x.tolist(), y.tolist())


# Loads in an image.
def load_image(date, file):
    file = 'Images/Radius/' + date + '-' + file + '.png'
    img = ndimage.imread(file, mode='L')
    return img


# Draws a celestial horizon
def draw_celestial_horizon(date, file):
    load_image(date, file)
    time = timestring_to_obj(date, file)

    dec = 0
    ra = 0
    while ra <= 360:
        xy = radec_to_xy(ra, dec, time)
        xy = (round(xy[0]), round(xy[1]))

        # Remember y is first, then x
        # Also make sure it's on the image at all.
        if xy[1] < 512 and xy[0] < 512:
            img[xy[1], xy[0]] = (244, 66, 229)

        ra += 0.5

    return img


# Draws a circle at the x y coord list with radius 5.
# Should probably note that right now it draws a square.
def draw_circle(x, y, img, color='c', name='blah.png'):
    # Generate Figure and Axes objects.
    figure = plot.figure()
    figure.set_size_inches(4, 4)  # 4 inches by 4 inches
    axes = plot.Axes(figure, [0., 0., 1., 1.])  # 0 - 100% size of figure

    # Turn off the actual visual axes for visual niceness.
    # Then add axes to figure
    axes.set_axis_off()
    figure.add_axes(axes)

    # Adds the image into the axes and displays it
    axes.imshow(img, cmap='gray')

    axes.set_aspect('equal')
    for i in range(0, len(x)):
        #circ = Circle((x[i], y[i]), 5, fill = False)
        circ = Rectangle((x[i]-5, y[i]-5), 11, 11, fill=False)
        circ.set_edgecolor(color)
        axes.add_patch(circ)

    plot.savefig(name, dpi=128)
    plot.close()


# Looks for a star in a variable size pixel box centered at x,y
# This just finds the "center of mass" for the square patch,
# with mass = greyscale value. With some modifications
# Mass is converted to exp(mass/10)
def find_star(img, centerx, centery, square=6):

    # We need to round these to get the center pixel as an int.
    centerx = np.int(round(centerx))
    centery = np.int(round(centery))
    # Just setting up some variables.
    R = (0, 0)
    M = 0

    # Half a (side length-1). i.e. range from x-square to x+square
    #square = 6

    # Fudge factor exists because I made a math mistake and somehow
    # it worked better than the correct mean.
    fudge = ((2 * square + 1)/(2 * square)) ** 2

    lowery = centery - square
    uppery = centery + square + 1
    lowerx = centerx - square
    upperx = centerx + square + 1

    temp = np.array(img[lowery: uppery, lowerx: upperx], copy=True)
    temp = temp.astype(np.float32)

    for x in range(0, len(temp[0])):
        for y in range(0, len(temp[0])):
            temp[y, x] = math.exp(temp[y, x]/10)

    averagem = np.mean(temp) * fudge
    #print(temp)
    # This is a box from -square to square in both directions
    # Range is open on the upper bound.
    for x in range(-square, square + 1):
        for y in range(-square, square + 1):
            m = img[(centery + y), (centerx + x)]

            m = math.exp(m/10)
            # Ignore the "mass" of that pixel
            # if it's less than the average of the stamp
            if m < averagem:
                m = 0
            #print(str(m) + ' ' + str(x) + ' ' + str(y))
            R = (m * x + R[0], m * y + R[1])
            M += m

    # Avoids divide by 0 errors.
    if M == 0:
        M = 1

    R = (R[0] / M, R[1] / M)
    star = (centerx + R[0], centery + R[1])

    # For some reason de-incrementing by 2 is more accurate than 1.
    # Don't ask me why, I don't understand it either.
    if square > 2:
        return find_star(img, star[0], star[1], square - 2)
    else:
        return star

    #print(R) # Debug
    #return (centerx + R[0], centery + R[1])


# Returns a tuple of the form (rexpected, ractual, deltar)
# Deltar = ractual - rexpected
def delta_r(img, centerx, centery):

    adjust1 = (centerx - center[0], center[1] - centery)

    rexpected = math.sqrt(adjust1[0] ** 2 + adjust1[1] ** 2)

    # If we think it's outside the circle then bail on all the math.
    # R of circle is 240, but sometimes the r comes out as 239.9999999
    if rexpected > 239:
        return (-1, -1, -1)

    # Put this after the bail out to save some function calls.
    star = find_star(img, centerx, centery)
    adjust2 = (star[0] - center[0], center[1] - star[1])

    ractual = math.sqrt(adjust2[0] ** 2 + adjust2[1] ** 2)
    deltar = ractual - rexpected

    return (rexpected, ractual, deltar)


#tempfile = 'r_ut043526s01920' #7/21
#tempfile = 'r_ut113451s29520' #7/31
#tempfile = 'r_ut035501s83760' #7/12
#tempfile = 'r_ut054308s05520' #7/19
#tempfile = 'r_ut063128s26700' #10/4 2016
date = '20170911'
tempfile = 'r_ut120511s41280'


# Polaris = 37.9461429,  89.2641378
# Sirius = 101.2875, -16.7161
# Vega = 279.235, 38.7837
# Arcturus = 213.915, 19.1822
# Alioth = 193.507, 55.9598
# Altair = 297.696, 8.86832
# Radec
stars = {'Polaris': (37.9461429,  89.2641378),
         'Altair': (297.696, 8.86832),
         'Vega': (279.235, 38.7837),
         'Arcturus': (213.915, 19.1822),
         'Alioth': (193.507, 55.9598),
         'Spica': (201.298, -11.1613),
         'Sirius': (101.2875, -16.7161)}

def contours(date, file):
    
    file = 'Images/' + date + '/' + file + '.png'

    img = ndimage.imread(file, mode='L')
    
    # Generate Figure and Axes objects.
    figure = plot.figure()
    figure.set_size_inches(4, 4)  # 4 inches by 4 inches
    axes = plot.Axes(figure, [0., 0., 1., 1.])  # 0 - 100% size of figure

    # Turn off the actual visual axes for visual niceness.
    # Then add axes to figure
    axes.set_axis_off()
    figure.add_axes(axes)

    # Adds the image into the axes and displays it
    axes.imshow(img, cmap='gray')

    axes.set_aspect('equal')
    
    for alt in range(0,100,30):
        r = np.interp(90 - alt, xp=thetapoints, fp=rpoints)
        r = r * 240 / 11.6  # mm to pixel rate
        
        circ = Circle(center, radius=r, fill=False, edgecolor='green')
        axes.add_patch(circ)
        
        
    name = "Images/test.png"
    plot.savefig(name, dpi=128)

    plot.close()

# Designed to test the conversion as well as the find_star method.
# Essentially all but useless now, but I may need to do so again in the future.
def conv_test():
    f = open('xy.txt', 'w')

    fileloc = 'Images/Radius/'
    files = os.listdir(fileloc)

    loc = 'Images/Find-Star/'

    mask = Mask.find_mask()
    for file in files:
        f.write(file + '\n')
        split = file.split('-')

        date = split[0]
        tempfile = split[1][:-4]

        img = load_image(date, tempfile)
        img = Mask.apply_mask(mask, img)
        xlist = []
        ylist = []

        xlist2 = []
        ylist2 = []

        # Assemble the list of expected star points
        for star in stars.keys():
            time = timestring_to_obj(date, tempfile)
            point = radec_to_xy(stars[star][0], stars[star][1], time)
            xlist.append(point[0])
            ylist.append(point[1])

        for i in range(0, len(xlist)):
            delta = delta_r(img, xlist[i], ylist[i])
            if delta != (-1, -1, -1):
                s = str(delta)[1:-1]

                # Expected
                altaz1 = xy_to_altaz(xlist[i], ylist[i])

                #point = find_star(img, xlist[i],ylist[i])

                x = xlist[i] - center[0]
                y = center[1] - ylist[i]
                point = galactic_conv(x, y, altaz1[1])

                point = (point[0] + center[0], center[1] - point[1])
                xlist2.append(point[0])
                ylist2.append(point[1])

                # Actual
                altaz2 = xy_to_altaz(point[0], point[1])

                deltaaz = altaz2[1] - altaz1[1]

                # Radius stuff
                #s = s + ', ' + str(altaz1[1]) + ', ' + str(altaz2[1]) + ', ' + str(deltaaz)

                # X-Y / Alt-Az stuff
                xy1 = str(point[0]) + ', ' + str(point[1])
                straltaz1 = str(altaz2[0]) + ', ' + str(altaz2[1])
                xy2 = str(xlist[i]) + ', ' + str(ylist[i])
                straltaz2 = str(altaz1[0]) + ', ' + str(altaz1[1])
                s = xy1 + ', ' + straltaz1 + ', ' + xy2 + ', ' + straltaz2

                f.write(s + '\n')
        f.write('\n')
        draw_circle(xlist, ylist, img, name=loc + date + '-1.png')
        draw_circle(xlist2, ylist2, img, color='y', name=loc + date + '-2.png')


    f.close()

#conv_test()

contours(date, tempfile)