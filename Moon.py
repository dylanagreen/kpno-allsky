import os
import math
import warnings

import ephem

import numpy as np

from scipy import ndimage

import matplotlib.pyplot as plt

import astropy.coordinates
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
import astropy.units as u
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from astropy.modeling.powerlaws import PowerLaw1D

import Coordinates
import ImageIO
import Histogram

from matplotlib.patches import Circle


# Radii in kilometers
# Radius of the earth is the radius of the earth shadow at the moon's orbit.
R_moon = 1737
R_earth = 4500

# Sets up a pyephem object for the camera.
camera = ephem.Observer()
camera.lat = '31.959417'
camera.lon = '-111.598583'
camera.elevation = 2120


# Returns the amount of the moon that is lit up still by the sun, and not
# shaded by the earth.
def eclipse_visible(d, R, r):

    # This makes addition work as addition and not concatenates
    d = np.asarray(d)
    d = np.abs(d)  # Required as for after totality times d < 0

    r2 = r * r
    R2 = R * R
    d2 = np.square(d)

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
    img = ndimage.imread('Images/Original/' + date + '/' + file, mode='L')
    thresh = 5
    img = np.where(img >= 255 - thresh, 1, 0)

    # Runs a closing to smooth over local minimums (which are mainly caused by
    # a rogue antenna). Then labels the connected white regions. Structure s
    # Makes it so that regions connected diagonally are counted as 1 region.
    s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    img = ndimage.morphology.binary_closing(img, structure=s)
    labeled, nums = ndimage.label(img, structure=s)

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

    return biggest


# Finds the x,y coordinates of the moon's center in a given image.
def find_moon(date, file):

    # Nicked this time formatting code from timestring to object.
    formatdate = date[:4] + '/' + date[4:6] + '/' + date[6:]
    time = file[4:6] + ':' + file[6:8] + ':' + file[8:10]
    formatdate = formatdate + ' ' + time

    # Sets the date of calculation.
    camera.date = formatdate

    # Calculates the sun and moon positions.
    moon = ephem.Moon()
    moon.compute(camera)

    # Conversion to x,y positions on the image.
    alt = np.degrees(moon.alt)
    az = np.degrees(moon.az)
    x, y = Coordinates.altaz_to_xy(alt, az)
    x, y = Coordinates.galactic_conv(x, y, az)

    return (x, y)


# Finds the x,y coordinates of the moon's center in a given image.
def find_sun(date, file):

    # Nicked this time formatting code from timestring to object.
    formatdate = date[:4] + '/' + date[4:6] + '/' + date[6:]
    time = file[4:6] + ':' + file[6:8] + ':' + file[8:10]
    formatdate = formatdate + ' ' + time

    # Sets the date of calculation.
    camera.date = formatdate

    # Calculates the sun and moon positions.
    sun = ephem.Sun()
    sun.compute(camera)

    # Conversion to x,y positions on the image.
    alt = np.degrees(sun.alt)
    az = np.degrees(sun.az)

    return (alt,az)


# Fits a Moffat fit to the moon and returns the estimated radius of the moon.
# Radius of the moon is the FWHM of the fitting function.
def fit_moon(img, x, y):

    # This block of code runs straight vertical from the center of the moon
    # It gives a predicted rough radius of the moon, it starts counting at the
    # first white pixel it encounters (the center may be black)
    # and stops at the last white pixel. White here defined as > 250 greyscale.
    yfloor = math.floor(y)
    count = False
    size = 0
    xfloor = math.floor(x)
    start = xfloor
    for i in range(0, 35):
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
                                 gamma=stddev)
    fit = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        model = fit(model_init, x, y, z)

    # /2 is average FWHM but FWHM = diameter, so divide by two again.
    #fwhm = (model.x_fwhm + model.y_fwhm) / 4
    fwhm = model.fwhm / 2

    return fwhm


# Generates the size vs illuminated fraction for the two eclipse nights.
def generate_eclipse_data(regen=False):

    dates = ['20180131', '20150404']

    # Function within a function to avoid code duplication.
    def data(date):
        # Necessary lists
        distances = []
        imvis = []
        truevis = []

        # Check to see if the data has been generated already.
        # If it has then read it from the file.
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

            # Nicked this time formatting code from timestring to object.
            formatdate = date[:4] + '/' + date[4:6] + '/' + date[6:]
            time = img[4:6] + ':' + img[6:8] + ':' + img[8:10]
            formatdate = formatdate + ' ' + time

            # This basically hacks us to use the center of the earth as our
            # observation point.
            camera.elevation = - ephem.earth_radius
            camera.date = formatdate

            # Calculates the sun and moon positions.
            moon = ephem.Moon()
            sun = ephem.Sun()
            moon.compute(camera)
            sun.compute(camera)

            # Finds the angular separation between the sun and the moon.
            sep = ephem.separation((sun.az, sun.alt), (moon.az, moon.alt))

            # Radius of moon orbit to convert angular separation -> distance
            R = 385000

            # For angles this small theta ~ sin(theta), so I dropped the sine
            # to save computation time.
            # Angle between moon and earth's shadow + angle between moon and sun
            # should ad d to pi, i.e. the earth's shadow is across from the sun.
            d = R * (np.pi - sep)

            size = moon_size(date, img)
            imvis.append(size)
            distances.append(d)

            print("Processed: " + date + '/' + img)

        # Calculates the proportion of visible moon for the given distance
        # between the centers.
        truevis = eclipse_visible(distances, R_earth, R_moon)

        imvis = np.asarray(imvis)

        # If the moon is greater than 40,000 pixels then I know that the moon
        # has merged with the light that comes from the sun and washes out the
        # horizon.
        imvis = np.where(imvis < 80000, imvis, float('NaN'))

        f = open(save, 'w')

        # Writes the data to a file so we can read it later for speed.
        for i in range(0, len(truevis)):
            f.write(str(truevis[i]) + ',' + str(imvis[i]) + '\n')
        f.close()

        return (truevis, imvis)

    trues = []
    ims = []
    for date in dates:
        true, im = data(date)
        trues.append(true)
        ims.append(im)

    return (trues, ims)


# Calculates the estimated size of the moon in the image based on the passed in
# illuminated fraction.
# Returns the radius of the circle that will cover the moon in the image.
def moon_circle(frac):
    illuminated = [0, 0.345, 0.71, 0.88, 0.97, 1.0]
    size = [650, 4000, 10500, 18000, 30000, 35000]

    A = np.interp(frac, illuminated, size)
    return np.sqrt(A/np.pi)


# Generates a mask that covers up the moon for a given image.
def moon_mask(date, file):
    # Get the fraction visible for interpolation and find the
    # location of the moon.
    vis = moon_visible(date, file)
    x,y = find_moon(date,file)

    # Creates the circle patch we use.
    r = moon_circle(vis)
    circ = Circle((x, y),r, fill=False)

    # The following code converts the patch to a 512x512 mask array, with True
    # for values outside the circle and False for those inside.
    # This is the same syntax as np.ma.make_mask returns.

    # This section of code generates an 262144x2 array of the
    # 512x512 pixel locations. 262144 = 512^2
    points = np.zeros((512**2,2))
    index = 0
    for i in range(0,512):
        for j in range(0,512):
            # These are backwards as expected due to how reshape works later.
            # Points is in x,y format, but reshape reshapes such that
            # it needs to be in y,x format.
            points[index,0] = j
            points[index,1] = i
            index += 1

    # Checks all the points are inside the circle, then reshapes it to the
    # 512x512 size.
    mask = circ.contains_points(points)
    mask = mask.reshape(512,512)

    return mask


# Generates a plot of illuminated fraction vs apparent moon size.
def generate_plots():
    # Loads the eclipse data
    vis, found = generate_eclipse_data()
    print("Eclipse data loaded!")

    # Eclipse normalization code.
    #found[0] = np.asarray(found[0]) / np.nanmax(found[0])
    #found[1] = np.asarray(found[1]) / np.nanmax(found[1])

    # Plots the two eclipses, the first in blue (default), the second in green
    plt.scatter(vis[0], found[0], label='2018/01/31 Eclipse', s=7)
    #plt.scatter(vis[1], found[1], label='2015/04/04 Eclipse', s=7, c='g')
    plt.ylabel("Approx Moon Size (pixels)")
    plt.xlabel("Illuminated Fraction")

    # Openthe file that tells us what images to use
    f1 = open("images.txt", 'r')

    # Vis is the portion of the moon illuminated by the sun that night
    # Found is the approximate size of the moon in the image
    vis = []
    found = []
    for line in f1:
        line = line.rstrip()
        info = line.split(',')
        vis.append(moon_visible(info[0], info[1]))
        found.append((moon_size(info[0], info[1] + '.png')))
        print("Processed: " + info[0] + '/' + info[1] + '.png')

    # Removes out any moons that appear too large in the images to be
    # considered valid.
    found = np.asarray(found)
    found = np.where(found < 40000, found, float('NaN'))

    # Normalizes the non eclipse data.
    #found1 = found / np.nanmax(found)

    # Adds the noneclipse data to the plot.
    plt.scatter(vis, found, label='Regular', s=7)

    # This plots the estimated model of moon size on top of the graph.
    vis2 = [0, 0.345, 0.71, 0.88, 0.97, 1.0]
    found2 = [650, 4000, 10500, 18000, 30000, 35000]
    plt.plot(vis2, found2, label='Model', c='r')

    # Interpolation estimate for the moon size in the image based on the
    # illuminated fractions.
    found3 = np.interp(vis, vis2, found2)
    plt.scatter(vis, found3, label='Interpolated', s=7)
    plt.legend()

    # Saves the figure, and then saves the same figure with a log scale.
    plt.savefig("Images/moon-size.png", dpi=256)

    ax = plt.gca()
    ax.set_yscale('log')
    plt.savefig("Images/moon-size-log.png", dpi=256)

    plt.close()


if __name__ == "__main__":
    #f1 = open("images.txt", 'r')

    date = '20170810'
    directory = 'Images/Original/' + date + '/'
    f1 = os.listdir(directory)
    f1 = sorted(f1)

    #for line in f1:
    for file in f1:
        #line = line.rstrip()
        #info = line.split(',')

        info = [None] * 2
        info[0] = date
        info[1] = file[:-4]

        path = 'Images/Original/' + info[0] + '/' + info[1]

        img = ndimage.imread(path + '.png', mode='L')

        mask = moon_mask(info[0], info[1])
        img1 = np.ma.masked_array(img, mask)


        bins,frac = Histogram.histogram(img, info[0] + '/' + info[1], mask)
        #Histogram.histogram(img, info[0] + '/' + info[1] + '-2.png')



        #ImageIO.save_image(cont, info[1], 'Images/Moontest', cmap='gray')


