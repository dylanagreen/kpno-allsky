from scipy import ndimage
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
import math
import os
import Mask
import ImageIO
import Coordinates

import astropy.coordinates
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.time.core as aptime
from astropy import units as u
from astropy.modeling import models, fitting
import warnings

from matplotlib.patches import Circle

center = (256, 252)

date = '20170718'

# Creates a histogram of the greyscale values in the image and saves it.
# Returns the histogram bin values.
def histogram(date, file):

    # Read in the image, then gets a mask
    img = ndimage.imread('Images/Original/' + date + '/' + file, mode = 'L')
    mask = Mask.generate_full_mask()
    mask = 1 - mask

    # Only want to histogram 0.3s images, as those are the ones with the moon.
    exposure = ImageIO.get_exposure(img)
    if exposure != 0.3:
        return (None, None)

    # Gets the location of the moon.
    moonx, moony = find_moon(date, file)
    r = fit_moon(img, moonx, moony)

    # Sets up the patch, radius = 5, color = cyan
    #circ = Circle((moonx, moony), 100, fill = False)
    #circ.set_edgecolor('c')

    # Converts the 1/0 array to True/False so it can be used as an index.
    # Then applies it, creating a new "image" array that only has the inside the
    # cicle items, but not the horizon items.
    mask = np.ma.make_mask(mask)
    img1 = img[mask]

    # Sets up the image so that the image is on the left and the histogram is
    # on the right.
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(10,5)

    # Turn off the actual visual axes on the image for visual niceness.
    # Then add the image to the left axes with the moon circle.
    ax[0].set_axis_off()
    ax[0].imshow(img, cmap='gray')
    #ax[0].add_patch(circ)

    # Creates the histogram with 256 bins (0-255) and places it on the right.
    bins = list(range(0,256))
    hist = ax[1].hist(img1.flatten(), bins = bins, color = 'blue', log = True)
    
    #plt.show()

    # Writes the moon radius onto the image.
    ax[0].text(0, -20, str(r), fontsize = 20)

    # Saving code.
    name = 'Images/Histogram/' + date + '/' + file
    if not os.path.exists('Images/Histogram/' + date + '/'):
        os.makedirs('Images/Histogram/' + date + '/')
    plt.savefig(name, dpi=256)
    print('Saved: ' + name)

    # Close the plot at the end.
    plt.close()
    
    # Return the histogram bin values in case you want to use it somewhere.

    return (hist[0], r)


# Finds the moon position and returns a the x,y position of the moon.
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


# Fits a gaussian to the moon and returns the estimated radius of the moon.
# Radius of the moon is the average of the FWHM in the x and y directions.
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


# Intializes the category defining histograms.
# Returns a dictionary where the key is the category number and the value is 
# the defining histogram for that category.
def init_categories():
    # Loads up the category numbers
    directory = 'Images/Category/'
    files = sorted(os.listdir(directory))
    categories = {}
    
    for file in files:
        
        # Opens the image, then uses np.histogram to generate the histogram
        # for that image, where the image is masked the same way as in the
        # histogram method.
        img = ndimage.imread('Images/Category/' + file, mode = 'L')
        mask = Mask.generate_full_mask()
        mask = 1 - mask

        mask = np.ma.make_mask(mask)
        img1 = img[mask]

        # Creates the histogram and adds it to the dict.
        bins = list(range(0,256))
        hist = np.histogram(img1, bins=bins)

        name = file[:-4]
        categories[name] = hist[0]
    
    return categories
    


# This method categorizes the histogram given based on the categories given.
# Categories should be a dict of categories, from init_categories for example.
# This method uses an algorithm called the histogram intersection algorithm.
def categorize(histogram, categories):
    
    best = 0
    category = None
    
    for cat in categories:
        
        # Take the minimum value of that bar from both histograms.
        minimum = np.minimum(histogram, categories[cat])
        
        # Then normalize based on the number of values in the category histogram
        # This is the intersection value.
        nummin = np.sum(minimum)
        numtot = np.sum(categories[cat])
        
        # Need to use true divide so the division does not floor itself.
        intersection = np.true_divide(nummin, numtot)
        
        # We want the category with the highest intersection value.
        if intersection > best:
            best = intersection
            category = cat
    
    # At present I'm currently looking for more categories, so if there isn't
    # a category with > thresh% intersection I want to know that.
    thresh = 0.35
    if best > thresh:
        print(best)
        return category
    else:
        return None
    

if __name__ == "__main__":

    # This code is here to loop through all currently downloaded dates.
    #dates = os.listdir('Images/Original')
    #dates.remove('.DS_Store')
    #dates.remove('1')
    #dates = sorted(dates)
    #print(dates)
    #for date in dates:

    #date = '20171108'
    #date = '20170817'
    date = '20161104'

    directory = 'Images/Original/' + date + '/'
    files = sorted(os.listdir(directory))

    cats = init_categories()
    
    saves = []
    
    for cat in cats:
        if not os.path.exists('Images/Histogram/' + date + '/' + cat + '/'):
            os.makedirs('Images/Histogram/' + date + '/' + cat + '/')
            
    if not os.path.exists('Images/Histogram/' + date + '/Moon/'):
        os.makedirs('Images/Histogram/' + date + '/Moon/')

    lowest = 100
    lowfile = ""
    for file in files:
        hist, r = histogram(date, file)
        #print("HIST")
        #print(hist)
        
        if hist is not None:
            newcat = categorize(hist, cats)
            print(newcat)
        else:
            newcat = None
        
        saves.append((file, newcat))
        
                
        if r is not None and 0 < r < lowest:
            lowfile = file
            lowest = r
            
    #print(lowfile)
    for loc in saves:
        file = loc[0]
        cat = loc[1]
    
        name1 = 'Images/Histogram/' + date + '/' + file
            
        if cat is not None and not file == lowfile:
            name2 = 'Images/Histogram/' + date + '/' + cat + '/' + file
            os.rename(name1, name2)
        elif cat is not None and file == lowfile:
            name2 = 'Images/Histogram/' + date + '/Moon/' + file
            os.rename(name1, name2)
