import warnings
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import astropy.coordinates
import astropy.time.core as aptime
import os
import math
import ImageIO
import Coordinates
from scipy import ndimage
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u
from astropy.modeling import models, fitting


# Radii in kilometers
# Radius of the earth is the radius of the earth shadow at the moon's orbit.
R_moon = 1737
R_earth = 4479

# Returns the amount of the moon that is lit up still by the sun, and not
# shaded by the earth.
def eclipse_visible(d, R, r):

    # This makes addition work as addition and not concatenates
    d = np.asarray(d)

    # Part 1 of the shaded area equation
    a = (d * d + r * r - R * R)
    b = d * 2 * r
    p1 = r * r * np.arccos(a / b)

    # Part 2 of the shaded area equation
    a = (d * d + R * R - r * r)
    b = d * 2 * R
    p2 = R * R * np.arccos(a / b)

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

    P = A / (np.pi * r * r)

    # P is the shaded area, so 1-P is the lit up area.
    P = 1 - P

    return P

# Calculates the proportion of the moon that is lit up for noneclipse nights.
# 1.0 = Full moon, 0.0 = New Moon
def moon_visible(date, file):
    
    # This is the first New Moon (P = 1.0) of 2018, and serves as a reference.
    ref = Time("2018-01-01 2:25:0")
    ref = ref.jd
    
    period = 29.530588
    
    time = Coordinates.timestring_to_obj(date, file)
    time = time.jd
    
    # Finds the passage of time, absolute value because the date might be before
    # the reference but we want the magnitude of the time passed.
    diff = abs(time - ref)
    delta = diff / period
    
    # Strips out only the fractional portion of the time change.
    phase = math.modf(delta)[0]
    
    # We start at a full moon, at 0.5 this will be 0, which is the new moon.
    # Then after 0.5 this works it's way back up to 1 for the next full moon.
    phase = abs(1.0 - 2 * phase)
    
    return phase
    

# Finds the size of the moon region (approximately) by taking pixels that are
# "close to white" (in this case, > 255 - threshold)
def moon_size(date, file):
    img = ndimage.imread('Images/Original/' + date + '/' + file, mode = 'L')

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

    # 0 = Start of moon being shaded
    # 1 = totality of eclipse
    eclipse_0 = (10) * 3600 + (49.8) * 60
    eclipse_1 = (12) * 3600 + (51.4) * 60

    # Time of this image in seconds
    time = int(file[4:6]) * 3600 + int(file[6:8]) * 60 + int(file[8:10])

    # Point slope form.
    slope = (R_moon + R_earth) / (eclipse_0 - eclipse_1)
    d = slope * (time - eclipse_1)

    # The moon is the biggest white area... most of the time.
    # TODO: Change this using a find moon function to just pick the region that
    # astropy says the moon is in.
    biggest = np.amax(sizes)

    return (biggest, d)

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

if __name__ == "__main__":
    date = '20180131'

    directory = 'Images/Original/' + date + '/'
    files = sorted(os.listdir(directory))

    for file in files:
        print(moon_visible(date, file))
