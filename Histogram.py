import numpy as np
import matplotlib.pyplot as plt
import os
import Mask
import ImageIO
import Moon
from copy import copy
from scipy import ndimage

center = (256, 252)

date = '20170718'


# Creates a histogram of the greyscale values in the image and saves it.
# Saves histogram to passed in path.
# Returns the histogram bin values.
def histogram(img, path, save=True):
    # Converts the 1/0 array to True/False so it can be used as an index.
    # Then applies it, creating a new "image" array that only has the inside the
    # cicle items, but not the horizon items.
    mask = Mask.generate_full_mask()
    mask = np.ma.make_mask(mask)
    img1 = np.ma.masked_array(img, mask)

    # Sets up the image so that the image is on the left and the histogram is
    # on the right.
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)

    # Turn off the actual visual axes on the image for visual niceness.
    # Then add the image to the left axes with the moon circle.
    ax[0].set_axis_off()

    # This new color palette is greyscale for all non masked pixels, and
    # red for any pixels that are masked and ignored.
    # Copied the old palette so I don't accidentally bugger it.
    palette = copy(plt.cm.gray)
    palette.set_bad('r', 1.0)
    ax[0].imshow(img1, cmap=palette)

    # Creates the histogram with 256 bins (0-255) and places it on the right.
    bins = list(range(0, 256))
    hist = ax[1].hist(img1.compressed(), bins=bins, color='blue', log=True)

    # Draws the vertical division line, in red
    thresh = 160
    bins = hist[0]

    # This slice ignores the white column, whereas the slice in total ignores
    # the black column.
    bins = bins[:-1]
    total = np.sum(bins[1:])
    clouds = np.sum(bins[thresh:])
    frac = clouds/total

    # Writes the fraction on the image
    ax[0].text(0, -20, str(frac), fontsize=20)

    ax[1].axvline(x=thresh, color='r')
    #plt.show()

    # Saving code.
    name = 'Images/Histogram/' + path

    # This ensures that the directory you're saving to actually exists.
    loc = path.rfind('/')
    dirname = 'Images/Histogram/' + path[0:loc]
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if save:
        plt.savefig(name, dpi=256)
        print('Saved: ' + name)

    # Close the plot at the end.
    plt.close()

    # Return the histogram bin values in case you want to use it somewhere.
    return (bins,frac)


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
        mask = Mask.generate_full_mask()
        mask = 1 - mask

        mask = np.ma.make_mask(mask)
        img1 = img[mask]

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
    dates = os.listdir('Images/Original')
    dates.remove('.DS_Store')
    dates.remove('1')
    dates = sorted(dates)

    date = '20180131'

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
        img = ndimage.imread('Images/Original/' + date + '/' + file, mode='L')
        hist = histogram(img, os.path.join(date, file))
        if hist is not None:
            newcat = categorize(hist, cats)
            print(newcat)
        else:
            newcat = None
        saves.append((file, newcat))

    for loc in saves:
        file = loc[0]
        cat = loc[1]

        name1 = 'Images/Histogram/' + date + '/' + file
        if cat is not None and not file == lowfile:
            name2 = 'Images/Histogram/' + date + '/' + cat + '/' + file
            #os.rename(name1, name2)
        elif cat is not None and file == lowfile:
            name2 = 'Images/Histogram/' + date + '/Moon/' + file
            #os.rename(name1, name2)
