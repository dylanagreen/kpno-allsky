import os
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import ndimage

import coordinates
import mask

center = (256, 252)

date = '20170718'

data = []
x = []
d1 = coordinates.timestring_to_obj('20171001', 'r_ut013603s08160').plot_date
d2 = coordinates.timestring_to_obj('20171031', 'r_ut132350s57840').plot_date


# Creates a histogram of the greyscale values in the image and saves it.
# Saves histogram to passed in path.
# Returns the histogram bin values.
# If a mask is passed, it uses that mask in addition to the one generated
# Mask
def plot_histogram(img, hist, masking, path, date, save=True):
    # Sets up the image so that the images are on the left
    # and the histogram and plot are on the right
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(10, 5.1)
    fig.subplots_adjust(hspace=.30, wspace=.07)

    # Turn off the actual visual axes on the images.
    ax[0, 0].set_axis_off()
    ax[1, 0].set_axis_off()

    # Display the original image underneath for transparency.
    ax[0, 0].imshow(img, cmap='gray')
    ax[1, 0].imshow(img, cmap='gray')

    # Creates the histogram with 256 bins (0-255) and places it on the right.
    # Kept this super general in case we want to change the amount of bins.
    bins = list(range(0, 255))
    width = 1

    ax[0, 1].bar(bins, hist, width=width, align='edge', color='blue', log=True)
    ax[0, 1].set_ylabel('Number of Occurrences')
    ax[0, 1].set_xlabel('Pixel Greyscale Value')

    # Cloudy pixels thresholded first, then the horizon and moon are masked.
    thresh = 160
    img2 = np.where(img >= thresh, 400, img)
    mask2 = mask.generate_full_mask()
    mask2 = np.ma.make_mask(mask2)
    img2 = np.ma.masked_array(img2, masking)
    img2 = np.ma.masked_array(img2, mask2)

    # This new color palette is greyscale for all non masked pixels, and
    # red for any pixels that are masked and ignored. It's blue for clouds.
    # Copied the old palette so I don't accidentally bugger it.
    palette = copy(plt.cm.gray)
    palette.set_bad('r', 0.5)
    palette.set_over('b', 0.5)

    # Need a new normalization so that blue pixels don't get clipped to white.
    ax[1, 0].imshow(img2, cmap=palette,
                    norm=colors.Normalize(vmin=0, vmax=255), alpha=1)

    # Writes the fraction on the image
    frac = cloudiness(hist)
    ax[0, 1].text(170, 2000, str(frac), fontsize=15, color='red')

    # Draws the vertical division line, in red
    ax[0, 1].axvline(x=thresh, color='r')

    # Forces the histogram to always have the same y axis height.
    ax[0, 1].set_ylim(1, 40000)

    data.append(frac)
    x.append(date)

    ax[1, 1].plot_date(x, data, xdate=True, markersize=1)
    ax[1, 1].set_ylim(0, 1.0)

    ax[1, 1].set_ylabel("Cloudiness Fraction")
    ax[1, 1].set_xlabel("Time after 01/01/2018")

    xt = []
    for i in range(20180101, 20180132):
        x1 = coordinates.timestring_to_obj(str(i), 'r_ut000000s00000').plot_date
        xt.append(x1)

    ax[1, 1].set_xticks(xt)
    ax[1, 1].set_xlim(xt[0], xt[-1])
    ax[1, 1].xaxis.grid(True)
    ax[1, 1].xaxis.set_ticklabels([])

    # Saving code.
    name = 'Images/Histogram/' + path

    # This ensures that the directory you're saving to actually exists.
    loc = path.rfind('/')
    dirname = 'Images/Histogram/' + path[0:loc]
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if save:
        dpi = 350.3
        plt.savefig(name, dpi=dpi, bbox_inches='tight')
        print('Saved: ' + name)

    # Close the plot at the end.
    plt.close()

    f = open('data.txt', 'a+')
    f.write(str(date) + ',' + str(frac) + '\n')
    f.close()


def generate_histogram(img, masking=None):
    # This first applies any passed in mask (like the moon mask)
    img1 = np.ma.masked_array(img, masking)

    # This then applies the horizon/circle mask.
    # Converts the 1/0 array to True/False so it can be used as an index.
    mask2 = mask.generate_full_mask()
    mask2 = np.ma.make_mask(mask2)
    img1 = np.ma.masked_array(img1, mask2)

    # Pixels from 0-255, so with 256 bins the histogram will give each pixel
    # value its own bin.
    bins = list(range(0, 256))
    hist, bins = np.histogram(img1.compressed(), bins)

    return (hist, bins)


def cloudiness(hist):
    # Pretty straight forward math here:
    # Num of pixels > thresh / total num of pixels.
    thresh = 160
    total = np.sum(hist)
    clouds = np.sum(hist[thresh:])
    frac = clouds/total

    return round(frac, 3)


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
        img = ndimage.imread('Images/Category/' + file, mode='L')
        masking = mask.generate_full_mask()
        masking = 1 - masking

        masking = np.ma.make_mask(masking)
        img1 = img[masking]

        # Creates the histogram and adds it to the dict.
        bins = list(range(0, 256))
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

        # Then normalize based on the num of values in the category histogram
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
    return None
