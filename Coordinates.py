import math
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.time.core as aptime
from astropy import units as u
import numpy as np

from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import matplotlib.image as image
import matplotlib.pyplot as plot
from scipy import ndimage

import os

import Mask

# Globals

# Center of the circle found using super acccurate photoshop layering techniques.
center = (256, 252)

# r - theta table.
rpoints = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 11.6]
thetapoints = [0, 3.58, 7.17, 10.76, 14.36, 17.98, 21.62, 25.27, 28.95, 32.66, 36.40, 40.17, 43.98, 47.83, 51.73, 55.67, 59.67, 63.72, 67.84, 72.03, 76.31, 80.69, 85.21, 89.97,90]

# Returns a tuple of form (alt, az)
def xy_to_altaz(x,y):

    # Point adjusted based on the center being at... well... the center. And not the top left. In case you were confused.
    # Y is measured from the top stop messing it up.
    pointadjust = (x - center[0], center[1] - y)

    # We use -x here because the E and W portions of the image are flipped
    az = math.atan2(-(pointadjust[0]),(pointadjust[1]))

    # For same reason as -x, we use > 0 here
    # atan2 ranges from -pi to pi so we have to shift the -pi to 0 parts up 2 pi.
    if pointadjust[0] > 0:
        az += 2 * math.pi

    az = math.degrees(az)

    # Pythagorean thereom boys.
    r = math.sqrt(pointadjust[0]**2 + pointadjust[1]**2)

    # For now if r is on the edge of the circle or beyond we'll have it just be 0 degrees. (Up from horizontal)
    if r > 240:
        alt = 0
    else:
        #r = 238 # Debug line
        r = r * 11.6 / 240 # Magic pixel to mm conversion rate

        # 90- turns the angle from measured from the vertical to measured from the horizontal.
        # This interpolates the value from the two on either side of it.
        alt = 90 - np.interp(r, xp = rpoints, fp = thetapoints)

    # Az correction
    az = az + .94444
    
    return (alt,az)

# Returns a tuple of form (ra, dec)
def altaz_to_radec(alt, az, time):
    assert type(time) is aptime.Time, "Time should be an astropy Time Object."

    # This is the latitutde/longitude of the camera
    cameraloc = (31.959417, -111.598583)

    cameraearth = EarthLocation(lat = cameraloc[0] * u.deg, lon = cameraloc[1] * u.deg, height = 2120 * u.m)

    altazcoord = SkyCoord(alt = alt * u.deg, az = az * u.deg, frame = 'altaz', obstime = time, location = cameraearth)
    radeccoord = altazcoord.icrs

    return (radeccoord.ra.degree, radeccoord.dec.degree)

# Returns a tuple of form (alt, az)
def radec_to_altaz(ra, dec, time):
    assert type(time) is aptime.Time, "Time should be an astropy Time Object."

    # This is the latitutde/longitude of the camera
    cameraloc = (31.959417, -111.598583)

    cameraearth = EarthLocation(lat = cameraloc[0] * u.deg, lon = cameraloc[1] * u.deg, height = 2120 * u.meter)


    # Creates the SkyCoord object
    radeccoord = SkyCoord(ra = ra, dec = dec, unit = 'deg', obstime = time, location = cameraearth, frame = 'icrs', temperature = 5 * u.deg_C, pressure = 78318 * u.Pa)

    # Transforms
    altazcoord = radeccoord.transform_to('altaz')

    return (altazcoord.alt.degree, altazcoord.az.degree)

# Returns a tuple of form (x,y)
def altaz_to_xy(alt, az):
    # Approximate correction (due to distortion of lens?)
    az = az - .94444

    #print(az)

    # Reverse of r interpolation
    r = np.interp(90 - alt, xp = thetapoints, fp = rpoints)

    r = r * 240 / 11.6 # mm to pixel rate

    # Remember angle measured from vertical so sin and cos are swapped from usual polar.
    # These are x,ys with respect to a zero.
    x = -1 * r * math.sin(math.radians(az))
    y = r * math.cos(math.radians(az))

    # y is measured from the top!
    pointadjust = (x + center[0], center[1] - y)

    return pointadjust

# Returns a tuple of form (x,y)
# Time must be an astropy Time object.
# Please.
def radec_to_xy(ra, dec, time):
    altaz = radec_to_altaz(ra, dec, time)
    x,y = altaz_to_xy(altaz[0], altaz[1])
    return galactic_conv(x,y,altaz[1])

# Returns a tuple of form (ra,dec)
# Time must be an astropy Time object.
# Please.
def xy_to_radec(x,y,time):
    altaz = xy_to_altaz(x,y)
    x,y = camera_conv(x,y,altaz[1])
    
    altaz = xy_to_altaz(x,y)
    return altaz_to_radec(altaz[0], altaz[1], time)

def timestring_to_obj(date, filename):
    # Add the dashes
    formatted = date[:4] + '-' + date[4:6] + '-' + date[6:]

    # Extracts the time from the file name. File names seem to be machine generated so this should not break.
    # Hopefully.
    time = filename[4:6] + ':' + filename[6:8] + ':' + filename[8:10]

    formatted = formatted + ' ' + time

    return Time(formatted)

# Converts from galactic x,y, expected az to camera x,y, actual az
def galactic_conv(x,y,az):

    r = math.sqrt(x**2 + y**2)
    az = az - .94444
    #print(r)
    # These magic numbers found using the genius method of "lots of trial and error."
    # Just kidding, I did a chi-squared analysis with a bunch of different models and this was the best I came up with.
    r = r + 2.369 * math.cos(math.radians(0.997 * (az - 42.088))) + 0.699
    az = az + 0.716 * math.cos(math.radians(1.015 * (az + 31.358))) - 0.181
    
    x = -1 * r * math.sin(math.radians(az))
    y = r * math.cos(math.radians(az))
    
    return (x,y)

# Converts from camera r,az to galactic r,az
def camera_conv(x,y,az):
    r = math.sqrt(x**2 + y**2)
    
    # You might think that this should be + but actually no. See next comment.
    az = az - .94444 #- 0.286375
    
    az = az - 0.731 * math.cos(math.radians(0.993 * (az + 34.5))) + 0.181
    r = r - 2.358 * math.cos(math.radians(0.99 * (az - 40.8))) - 0.729
    
    x = -1 * r * math.sin(math.radians(az))
    y = r * math.cos(math.radians(az))
    
    return (x,y)

# Draws a celestial horizon
def celestialhorizon(date, file):
    loadimage(date,file)
    time = timestring_to_obj(date,file)

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

# Loads in an image.
def loadimage(date, file):
    time = timestring_to_obj(date, file)

    file = 'Images/Radius/' + date + '-' + file + '.png'

    img = ndimage.imread(file, mode = 'L')

    return img

# Draws a circle at the x y coord list with radius 5.
def circle(x,y,img, color = 'c', name = 'blah.png'):


    # Generate Figure and Axes objects.
    figure = plot.figure()
    figure.set_size_inches(4,4) # 4 inches by 4 inches
    axes = plot.Axes(figure,[0.,0.,1.,1.]) # 0 - 100% size of figure

    # Turn off the actual visual axes for visual niceness.
    # Then add axes to figure
    axes.set_axis_off()
    figure.add_axes(axes)

    # Adds the image into the axes and displays it
    axes.imshow(img, cmap = 'gray')

    axes.set_aspect('equal')
    for i in range(0,len(x)):
        #circ = Circle((x[i], y[i]), 5, fill = False)
        circ = Rectangle((x[i]-5, y[i]-5), 11, 11, fill = False)
        circ.set_edgecolor(color)
        axes.add_patch(circ)

    # DPI chosen to have resultant image be the same size as the originals. 128*4 = 512
    plot.savefig(name, dpi = 128)

    plot.close()
    # Show the plot
    #plot.show()

# Looks for a star in a variable size pixel box centered at x,y
# This just finds the "center of mass" for the square patch, with mass = greyscale value. With some modifications
# Mass is converted to exp(mass/10)
def findstar(img, centerx, centery, square = 6):

    # We need to round these to get the center pixel as an int.
    centerx = np.int(round(centerx))
    centery = np.int(round(centery))
    # Just setting up some variables.
    R = (0, 0)
    M = 0

    # Half a (side length-1). i.e. range from x-sqare to x+square
    #square = 6

    # Fudge factor exists because I made a math mistake and somehow it worked better than the correct mean.
    # Er, I mean, this is totally legit math here.
    fudge = ((2 * square + 1)/(2 * square)) ** 2
    temp = np.array(img[centery - square : centery + square + 1, centerx - square : centerx + square + 1], copy = True)
    temp = temp.astype(np.float32)

    for x in range(0, len(temp[0])):
        for y in range(0, len(temp[0])):
            temp[y,x] = math.exp(temp[y,x]/10)

    averagem = np.mean(temp) * fudge
    #print(temp)
    # This is a box from -square to square in both directions, range is open on the upper bound remember?
    for x in range(-square, square + 1):
        for y in range(-square, square + 1):
            m = img[(centery + y), (centerx + x)]
            
            m = math.exp(m/10)
            # Ignore the "mass" of that pixel if it's less than the average of the stamp
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
    
    # For some reason incrementing by 2 is more accurate than 1. Don't ask me why, I don't understand it either.
    if square > 2:
        return findstar(img, star[0], star[1], square-2)
    else:
        return star

    #print(R) # Debug
    #return (centerx + R[0], centery + R[1])

# Returns a tuple of the form (rexpected, ractual, deltar)
# Deltar = ractual - rexpected
def deltar(img, centerx, centery):

    adjust1 = (centerx - center[0], center[1] - centery)

    rexpected = math.sqrt(adjust1[0]**2 + adjust1[1] ** 2)

    # If we think it's outside the circle then screw this lol.
    # R of circle is 240, but sometimes the r comes out as 239.9999999 so just do > 239
    if rexpected > 239:
        return (-1,-1,-1)

    # Put this after the bail out to save some function calls.
    star = findstar(img, centerx, centery)
    adjust2 = (star[0] - center[0], center[1] - star[1])

    ractual = math.sqrt(adjust2[0]**2 + adjust2[1] ** 2)
    deltar = ractual - rexpected

    return (rexpected, ractual, deltar)

altaz = xy_to_altaz(250,300)

#tempfile = 'r_ut043526s01920' #7/21
#tempfile = 'r_ut113451s29520' #7/31
#tempfile = 'r_ut035501s83760' #7/12
#tempfile = 'r_ut054308s05520' #7/19
#tempfile = 'r_ut063128s26700' #10/4 2016
date = '20170712'
tempfile = 'r_ut035501s83760'

#print(altaz_to_radec(altaz[0],altaz[1],timestring_to_obj('20170719', tempfile)))

#star = SkyCoord.from_name('Siruis', 'icrs')
#print(star)

# Polaris = 37.9461429,  89.2641378
# Sirius = 101.2875, -16.7161
# Vega = 279.235, 38.7837
# Arcturus = 213.915, 19.1822
# Alioth = 193.507, 55.9598
#'Altair' : (297.696, 8.86832),
# Radec
stars = {'Polaris' : (37.9461429,  89.2641378), 'Altair' : (297.696, 8.86832), 'Vega'  : (279.235, 38.7837), 'Arcturus' : (213.915, 19.1822), 'Alioth' : (193.507, 55.9598), 'Spica' : (201.298, -11.1613), 'Sirius' : (101.2875, -16.7161)}


xlist = []
ylist = []

# Assemble a list of the points to circle.
for star in stars.keys():
    #print(star)
    point = radec_to_xy(stars[star][0], stars[star][1], timestring_to_obj(date, tempfile))
    #point = xy_to_altaz(stars[star][0],stars[star][1])
    #print(str(point) + '\n')
    xlist.append(point[0])
    ylist.append(point[1])


# Designed to test the conversion as well as the find_star method. 
# Essentially all but useless now, but I may need to do so again in the future.
def conv_test():
    f = open('xy.txt', 'w')

    fileloc = 'Images/Radius/'
    files = os.listdir(fileloc)

    loc = 'Images/Find-Star/'
    j = 1
    mask = Mask.findmask()
    for file in files:
        f.write(file + '\n')
        split = file.split('-')
        
        date = split[0]
        tempfile = split[1][:-4]
        
        img = loadimage(date, tempfile)
        img = Mask.applymask(mask, img)
        xlist = []
        ylist = []
        
        
        xlist2 = []
        ylist2 = []
        
        # Assemble the list of expected star points
        for star in stars.keys():
            point = radec_to_xy(stars[star][0], stars[star][1], timestring_to_obj(date, tempfile))
            xlist.append(point[0])
            ylist.append(point[1])

        for i in range(0,len(xlist)):
            delta = deltar(img, xlist[i],ylist[i])
            if delta != (-1,-1,-1):
                s = str(delta)[1:-1]

                # Expected
                altaz1 = xy_to_altaz(xlist[i],ylist[i])
                
                #point = findstar(img, xlist[i],ylist[i])
                
                x = xlist[i] - center[0]
                y = center[1] - ylist[i]
                point = galactic_conv(x, y, altaz1[1])
                
                point = (point[0] + center[0], center[1] - point[1])
                xlist2.append(point[0])
                ylist2.append(point[1])

                # Actual
                altaz2 = xy_to_altaz(point[0], point[1])

                

                deltaaz = altaz2[1]-altaz1[1]
                
                # Radius stuff
                #s = s + ', ' + str(altaz1[1]) + ', ' + str(altaz2[1]) + ', ' + str(deltaaz)
                
                # X-Y / Alt-Az stuff
                s = str(point[0]) + ', ' + str(point[1]) + ', ' + str(altaz2[0]) + ', ' + str(altaz2[1]) + ', ' + str(xlist[i]) + ', '+ str(ylist[i]) + ', ' + str(altaz1[0]) + ', ' + str(altaz1[1])
                
                f.write(s + '\n')
        f.write('\n')
        circle(xlist,ylist,img, name = loc + date + '-1.png')
        circle(xlist2,ylist2,img,color = 'y', name = loc + date + '-2.png')
        j += 1

    f.close()
