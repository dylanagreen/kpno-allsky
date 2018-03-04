import os
import math
import warnings

import ephem

import numpy as np

from scipy import ndimage

import matplotlib.image as image
import matplotlib.pyplot as plt

import astropy.coordinates
import astropy.time.core as aptime
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from astropy.modeling import models,fitting

import ImageIO
import Coordinates


# Radii in kilometers
# Radius of the earth is the radius of the earth shadow at the moon's orbit.
R_moon = 1737
R_earth = 4479

# Returns the amount of the moon that is lit up still by the sun, and not
# shaded by the earth.
def eclipse_visible(d, R, r):

    # This makes addition work as addition and not concatenates
    d = np.asarray(d)
    d = np.abs(d) # Required as for after totality times d < 0

    r2 = r * r
    R2 = R * R
    d2 = np.square(d)

    #d1 = d * d - r2 + R2

    # Part 1 of the shaded area equation
    a = (d2 + r2 - R2)
    b = d * 2 * r
    p1 = r2 * np.arccos(a / b)

    # Part 2 of the shaded area equation
    a = (d2 + R2 - r2)
    b = d * 2 * R
    p2 = R2 * np.arccos(a / b)

    # Part 3 of the shaded area equation
    a1 = (r + R - d)
    a2 = (d + r - R)
    a3 = (d - r + R)
    a4 = (d + r + R)
    p3 = (0.5) * np.sqrt(a1 * a2 * a3 * a4)

    # Add them together to get the shaded area
    A = p1 + p2 - p3

    # Get the shaded proportion by divding the shaded area by the total area
    # Assumes r is the radius of the moon being shaded.
    P = A / (np.pi * r2)

    # P is the shaded area, so 1-P is the lit up area.
    P = 1 - P

    return P

# Calculates the proportion of the moon that is lit up for noneclipse nights.
# 1.0 = Full moon, 0.0 = New Moon
def moon_visible(date, file):

    #phase = (1/2) * math.cos(2 * math.pi * diff / period) + (1/2)

    # Nicked this time formatting code from timestring to object.
    formatdate = date[:4] + '/' + date[4:6] + '/' + date[6:]
    time = file[4:6] + ':' + file[6:8] + ':' + file[8:10]
    formatdate = formatdate + ' ' + time

    # Sets up a pyephem object for the camera.
    camera = ephem.Observer()
    camera.lat = '31.959417'
    camera.lon = '-111.598583'
    camera.elevation = 2120
    camera.date = formatdate

    # Makes a moon object and calculates it for the observation location/time
    moon = ephem.Moon()
    moon.compute(camera)

    return moon.moon_phase


# Finds the size of the moon region (approximately) by taking pixels that are
# "close to white" (in this case, > 255 - threshold)
def moon_size(date, file):
    img = ndimage.imread('Images/Original/' + date + '/' + file, mode = 'L')
    img1 = np.copy(img)

    thresh = 5
    img = np.where(img >= 255 - thresh, 1, 0)

    # Runs a closing to smooth over local minimums (which are mainly caused by
    # a rogue antenna). Then labels the connected white regions. Structure s
    # Makes it so that regions connected diagonally are counted as 1 region.
    s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    img = ndimage.morphology.binary_closing(img, structure = s)
    labeled, nums = ndimage.label(img, structure = s)

    # Want to find the size of each labeled region.
    sizes = [0] * (nums + 1)
    for row in labeled:
        for val in row:
            sizes[val] = sizes[val] + 1

    # We want to exclude the background from the "biggest region" calculation.
    # It's quicker to just set the background (0) region to 0 than to check
    # every single value above before adding it to the array.
    sizes[0] = 0

    # Following code calculates d, the distance between the center of
    # Earth's shadow and the center of the moon. Basically just d = v*t.

    # Use astropy to find the labeled region that the moon is in.
    posx, posy = find_moon(date, file)
    posx = math.floor(posx)
    posy = math.floor(posy)

    reg = labeled[posy, posx]

    # Very large and bright moons have a dark center (region 0) but I want the
    # region of the moon.
    while reg == 0 and posx < 511:
        posx = posx + 1
        reg = labeled[posy, posx]


    biggest = sizes[reg]

    #temp = np.where(labeled == reg, 1, 0)
    #ImageIO.save_image(temp, file + '-1', 'Images/Temp/' + date, cmap='gray')
    #ImageIO.save_image(img1, file + '-2', 'Images/Temp/' + date, cmap='gray')

    return biggest

# Finds the x,y coordinates of the moon's center in a given image.
def find_moon(date, file):

    # Sets up location and time variables.
    time = Coordinates.timestring_to_obj(date, file)
    camera = (31.959417 * u.deg, -111.598583 * u.deg)
    cameraearth = EarthLocation(lat=camera[0], lon=camera[1],
                                height=2120 * u.meter)

    # Gets a SkyCoord with the moon position.
    moon = astropy.coordinates.get_moon(time, cameraearth)
    #print(moon.frame)

    # Converstion to x,y positions on the image.
    altazcoord = moon.transform_to(AltAz(obstime=time, location=cameraearth))
    alt = altazcoord.alt.degree
    az = altazcoord.az.degree
    x, y = Coordinates.altaz_to_xy(alt, az)
    x, y = Coordinates.galactic_conv(x, y, az)

    return (x,y)



# Fits a Moffat fit to the moon and returns the estimated radius of the moon.
# Radius of the moon is the FWHM of the fitting function.
def fit_moon(img, x, y):

    # This block of code runs straight vertical from the center of the moon
    # It gives a predicted rough radius of the moon, it starts counting at the
    # first white pixel it encounters (because the center may be overflow black)
    # and stops at the last white pixel. White here defined as > 250 greyscale.
    yfloor = math.floor(y)
    count = False
    size = 0
    xfloor = math.floor(x)
    start = xfloor
    for i in range(0,35):
        start += 1

        # Breaks if it reaches the edge of the image.
        if start == img.shape[1]:
            break
        if not count and img[yfloor, start] >= 250:
            count = True
        elif count and img[yfloor, start] >= 250:
            size += 1
        elif count and img[yfloor, start] < 250:
            break

    # Add some buffer pixels in case the center is black and the edges of the
    # moon are fuzzed.
    size = (size + 10) * 2

    # Makes sure the lower/upper slices don't out of bounds error.
    lowerx = xfloor - size if (xfloor - size > 0) else 0
    lowery = yfloor - size if (yfloor - size > 0) else 0
    upperx = xfloor + size if (xfloor + size < 511) else 511
    uppery = yfloor + size if (yfloor + size < 511) else 511

    # Size of the moon enclosing square.
    deltax = (upperx - lowerx)
    deltay = (uppery - lowery)

    # Creates two arrays, with the array values being the x or y coordinate of
    # that location in the array.
    y, x = np.mgrid[0:deltay, 0:deltax]

    # Slices out the moon square and finds center coords.
    z = img[lowery:uppery, lowerx:upperx]
    midy = deltay / 2
    midx = deltax / 2

    # Moffat fit, centered in square, stdev of 20 as a start.
    stddev = 20
    model_init = models.Moffat2D(amplitude=200, x_0=midx, y_0=midy,
                                 gamma = stddev)
    fit = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        model = fit(model_init, x, y, z)

    # /2 is average FWHM but FWHM = diameter, so divide by two again.
    #fwhm = (model.x_fwhm + model.y_fwhm) / 4
    fwhm = model.fwhm / 2

    return fwhm


def generate_eclipse_data(regen = False):

    dates = ['20180131', '20150404']

    eclipsestart = {'20180131':(11) * 3600 + (47.6) * 60,
                    '20150404':(10) * 3600 + (16) * 60}
    eclipsetotal = {'20180131':(12) * 3600 + (51.4) * 60,
                    '20150404':(11) * 3600 + (58) * 60}

    # 0 = Start of moon being shaded
    # 1 = totality of eclipse
    def data(date, eclipse_0, eclipse_1):
        # Necessary lists
        distances = []
        imvis = []
        truevis = []

        # Check to see if the data has been generated already. If it has then read
        # it from the file
        save = 'eclipse-' + date + '.txt'
        if os.path.isfile(save) and not regen:
            f = open(save)
            for line in f:
                line = line.rstrip().split(',')
                truevis.append(float(line[0]))
                imvis.append(float(line[1]))
            f.close()
            return (truevis, imvis)

        # If we're regenerating the data we do it here.

        directory = 'Images/Original/' + date + '/'
        images = sorted(os.listdir(directory))

        # Finds the size of the moon in each image.
        for img in images:

            # Time of this image in seconds
            time = int(img[4:6]) * 3600 + int(img[6:8]) * 60 + int(img[8:10])

            # Point slope form. Totality occurs when d = R_e - R_m, and not at
            # d = 0 as originally assumed.
            slope = (2 + R_moon) / (eclipse_0 - eclipse_1)
            d = slope * (time - eclipse_1) + (R_earth - R_moon)

            size = moon_size(date, img)
            imvis.append(size)
            distances.append(d)

        # Calculates the proportion of visible moon for the given distance between
        # the centers.
        truevis = eclipse_visible(distances, R_earth, R_moon)

        imvis = np.asarray(imvis)

        # If the moon is greater than 40,000 pixels then I know that the moon
        # has merged with the light that comes from the sun and washes out the
        # horizon.
        imvis = np.where(imvis < 80000, imvis, float('NaN'))

        f = open(save, 'w')

        for i in range(0,len(truevis)):
            f.write(str(truevis[i]) + ',' + str(imvis[i]) + '\n')
        f.close()

        return (truevis, imvis)

    trues = []
    ims = []
    for date in dates:
        true, im = data(date, eclipsestart[date], eclipsetotal[date])
        trues.append(true)
        ims.append(im)

    return (trues, ims)


if __name__ == "__main__":

    vis, found = generate_eclipse_data()
    print("Eclipse data loaded!")

    # Some fitting code for later.
    #t_init = models.Sine1D()
    #fit_t = fitting.LevMarLSQFitter()
    #t = fit_t(t_init, found, vis)

    plt.scatter(vis[0], found[0], label='2018/01/31 Eclipse', s=7)
    plt.ylabel("Approx Moon Size (pixels)")
    plt.xlabel("Proportion of moon visible")

    plt.scatter(vis[1], found[1], label='2015/04/04 Eclipse', s=7, c='g')

    #plt.plot(t(vis), vis)

    f1 = open("images.txt", 'r')

    vis = []
    found = []
    for line in f1:
        line = line.rstrip()
        info = line.split(',')
        vis.append(moon_visible(info[0], info[1]))
        found.append((moon_size(info[0], info[1] + '.png')))
        print("Processed: " + info[0] + '/' + info[1] + '.png')

    found = np.asarray(found)
    found = np.where(found < 40000, found, float('NaN'))
    print(vis)
    print(found)
    plt.scatter(vis, found, label='Regular', s=7)
    plt.legend()

    #ax = plt.gca()
    #ax.set_yscale('log')

    plt.savefig("Images/moon-size.png", dpi=256)

    #plt.show()
