import math
import os
import numpy as np
from scipy import ndimage

from astropy.coordinates import SkyCoord, EarthLocation, get_moon, AltAz
from astropy import units as u
from matplotlib.patches import Circle
import matplotlib.pyplot as plot

import Mask
import ImageIO
import Coordinates


center = (256, 252)

def zero_three_cloud_contrast(img, name, date):
    img2 = ndimage.imread('Images/Original/' + date + '/r_ut052936s31200.png', mode='L')
    
    img3 = np.copy(img)
    img = np.int16(img)
    img2 = np.int16(img2)
    
    
    val = img[510,510] - img2[510,510]
    img = img - val
    
    test = ImageIO.image_diff(img, img2)
    loc = 'Images/Cloud/' + str(date) + ' - 3'
    
    test = ndimage.grey_closing(test, size = (2,2))
    
    avg = np.mean(test)
    
    cond = np.where(test > avg, 0, 1)
    
    final = np.multiply(img3, cond)
    
    # Find the mask and black out those pixels.
    mask = Mask.generate_mask()
    final = Mask.apply_mask(mask, final)
    
    ImageIO.save_image(test, name, loc, 'gray')

def zero_three_cloud_contrast2(img, name, date):
    
    dpi = 128
    y = img.shape[0] / dpi
    x = img.shape[1] / dpi
    
    # Generate Figure and Axes objects.
    figure = plot.figure()
    figure.set_size_inches(x, y)  # 4 inches by 4 inches
    axes = plot.Axes(figure, [0., 0., 1., 1.])  # 0 - 100% size of figure

    # Turn off the actual visual axes for visual niceness.
    # Then add axes to figure
    axes.set_axis_off()
    figure.add_axes(axes)

    # Adds the image into the axes and displays it
    # Then saves
    axes.imshow(img, cmap = 'gray')
    
    # This is the latitude/longitude of the camera
    camera = (31.959417 * u.deg, -111.598583 * u.deg)

    cameraearth = EarthLocation(lat=camera[0], lon=camera[1],
                                height=2120 * u.meter)
    
    time = Coordinates.timestring_to_obj(date, name)
    
    #print(time)
    moon = get_moon(time, location = cameraearth)
    
    moonaltaz = moon.transform_to(AltAz(obstime = time, location = cameraearth))

    alt = moonaltaz.alt.degree
    az =  moonaltaz.az.degree
    x, y = Coordinates.altaz_to_xy(alt, az)
    
    pos = Coordinates.galactic_conv(x, y, az)
    
    r = 70
    circ = Circle(pos, radius=r, fill=False, edgecolor='green')
    axes.add_patch(circ)
    
    location = 'Images/Cloud/' + str(date) + '/'
    name = location + name
    
    # Print "saved" after saving, in case saving messes up.
    plot.savefig(name, dpi=dpi)
    print('Saved: ' + name)

    # Show the plot
    #plot.show()

    # Close the plot in case you're running multiple saves.
    plot.close()


# Takes in an image as an np ndarray.
# Name and date are for saving purposes
# TODO Refactor to return image, save elsewhere
def six_cloud_contrast(img, name, date):

    # Find the mask and black out those pixels.
    mask = Mask.generate_mask()
    img = Mask.apply_mask(mask, img)

    # Inverts and subtracts 4 * the original image. This replicates the previous
    # behaviour in one step.
    # Previous work flow: Invert, subtract, subtract, subtract.
    # If it goes negative I want it to be 0 rather than positive abs of num.
    invert = 255 - 4 * np.int16(img)
    invert = np.where(invert < 0, 0, invert)

    # Smooth out the black holes left where stars were in the original.
    # We need them to be "not black" so we can tell if they're in a region.
    closedimg = ndimage.grey_closing(invert, size=(2,1))

    # Thresholds the image into black and white with a value of 10.
    # Pixels brighter than greyscale 10 are white, less than are 0.
    binimg = np.where(closedimg > 10, 1, 0)

    # Cleans up "floating" white pixels.
    binimg = ndimage.binary_opening(binimg)

    # Mask out the horizon objects so they don't mess with cloud calculations.
    binimg = Mask.apply_mask(mask, binimg)

    # Expand the white areas to make sure they cover the items they represent
    # from the inverted image.
    binimg = ndimage.binary_dilation(binimg)

    # Creates a buffer circle keeping the image isolated from the background.
    for row in range(0,img.shape[1]):
        for column in range(0,img.shape[0]):
            x = column - center[0]
            y = center[1] - row
            r = math.sqrt(x**2 + y**2)
            if (r < 246) and (r > 241):
                binimg[row,column] = 0


    # This structure makes it so that diagonally connected pixels are part of
    # the same region.
    struct = [[True, True, True],[True, True, True],[True, True, True]]
    labeled, num_features = ndimage.label(binimg, structure = struct)
    regionsize = [0] * (num_features + 1)
    starnums = [0] * (num_features + 1)

    for row in range(0,img.shape[1]):
        for column in range(0,img.shape[0]):
            regionsize[labeled[row,column]] += 1

            # This finds stars in "cloud" regions
            # Basically, if somewhat bright, and the region is marked "cloud."
            if img[row, column] >= (95) and binimg[row, column] == 1:
                x = column - center[0]
                y = center[1] - row
                r = math.sqrt(x**2 + y**2)
                if(r <= 240):
                    regionnum = labeled[row, column]
                    starnums[regionnum] += 1

    # The reason why I use density is mainly because of very small non-clouds.
    # They contain few stars, which rules out a strictly star count method.
    # This, however, is actually density^-1. I.e. it's size/stars rather than
    # stars/size. This is because stars/size is very small sometimes.
    # I'm aware of a division by 0 warning here. If a region has no stars, then
    # this divides by 0. In fact this np.where exists to ignore that and set
    # zero star regions to a density of 0, since I ignore those later.
    # Hence I'm supressing the divide by 0 warning for these two lines.
    with np.errstate(divide='ignore'):
        density = np.divide(regionsize, starnums)
        density = np.where(np.asarray(starnums) < 1, 0, density)

    # Zeroes out densities < 12
    density = np.where(density < 12, 0, density)
    density[0] = 350

    # Creates a density "image".
    # This is an image where each feature has its value set to its density.
    for row in range(0,labeled.shape[1]):
        for column in range(0,labeled.shape[0]):
            value = labeled[row, column]
            labeled[row, column] = density[value]


    # If the value is less than the mean density, we want to mask it in the
    # "map" image. Hence set it to 0, everything else to 1, and multipy.
    # This keeps the non masks (x*1 = x) and ignores the others (x*0 = 0)
    m = np.mean(density[np.nonzero(density)])
    masked = np.where(labeled < m, 0, 1)
    invert2 = np.multiply(invert, masked)


    # The thinking here is that the whiter it is in the contrast++ image, the
    # darker it should be in the original. Thus increasing cloud contrast
    # without making it look like sketchy black blobs.
    multiple = .6 - invert2 / 255
    newimg = np.multiply(img, multiple)

    loc = 'Images/Cloud/' + str(date)
    ImageIO.save_image(newimg, name, loc, 'gray')


if __name__ == "__main__":
    date = '20171108'
    directory = 'Images/Original/' + date + '/'
    files = os.listdir(directory)
    #file = 'r_ut105647s18240.png'
    #file = 'r_ut080515s07920.png'
    #file = 'r_ut113241s20400.png'
    for file in files:
        img = ndimage.imread(directory + file, mode='L')
        zero_three_cloud_contrast(img, file, date)



