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

from matplotlib.patches import Circle

center = (256, 252)

date = '20170718'

# Creates a histogram of the greyscale values in the image and saves it.
def histogram(date, file):

    # Read in the image, then gets a mask
    img = ndimage.imread('Images/Original/' + date + '/' + file, mode = 'L')
    mask = Mask.generate_full_mask()
    mask = 1 - mask

    # Only want to histogram 0.3s images, as those are the ones with the moon.
    exposure = ImageIO.get_exposure(img)
    if exposure != 0.3:
        return

    # Gets the location of the moon.
    circ = find_moon(img, date, file)

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
    ax[0].add_patch(circ)

    # Creates the histogram with 256 bins (0-255) and places it on the right.
    bins = list(range(0,256))
    ax[1].hist(img1.flatten(), bins = bins, color = 'blue', log = True)
    #plt.show()

    # Saving code.
    name = 'Images/Histogram/' + date + '/' + file
    if not os.path.exists('Images/Histogram/' + date + '/'):
        os.makedirs('Images/Histogram/' + date + '/')
    plt.savefig(name, dpi=256)
    print('Saved: ' + name)

    # Close the plot at the end.
    plt.close()


# Finds the moon and returns a circle at that location
def find_moon(image, date, file):

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

    # Sets up the patch, radius = 5, color = cyan
    circ = Circle((x, y), 5, fill = False)
    circ.set_edgecolor('c')

    return circ

#dates = os.listdir('Images/Original')
#dates.remove('.DS_Store')
#dates.remove('1')
#dates = sorted(dates)
#print(dates)
#for date in dates:

date = '20160823'

directory = 'Images/Original/' + date + '/'
files = sorted(os.listdir(directory))

for file in files:
    histogram(date, file)

#test('r_ut012616s16560.png')
