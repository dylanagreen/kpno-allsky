import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import os
from scipy import ndimage

import ImageIO

# This is necessary.
# Python works on tuples as required for color medians, which means we need to
# turn the numpy color array of length 3 into a tuple.
def ndarray_to_tuplelist(arr):
    templist = []

    # Runs over the second dimension (the longer one)
    for i in range(0, arr.shape[1]):
        tup = (arr[0, i], arr[1, i], arr[2, i], arr[3, i])
        templist.append(tup)

    return templist


# This works as wanted for tuples yay!
def median_of_medians(arr, i):

    # Divide the array into sublists of length 5 and find the medians.
    sublists = []
    medians = []

    for j in range(0, len(arr), 5):
        temp = arr[j:j+5]
        sublists.append(temp)

    for sublist in sublists:
        medians.append(sorted(sublist)[len(sublist)//2])

    if len(medians) <= 5:
        pivot = sorted(medians)[len(medians)//2]
    else:
        # Find the median of the medians array using this method.
        pivot = median_of_medians(medians, len(medians)//2)

    low = [j for j in arr if j < pivot]
    high = [j for j in arr if j > pivot]
    identicals = [j for j in arr if j == pivot]

    lownum = len(low)
    # This edit is required to make sure this is valid for lists with dupes.
    identnum = len(identicals)

    if i < lownum:
        return median_of_medians(low, i)
    elif i < identnum + lownum:
        return pivot
    else:
        return median_of_medians(high, i - (lownum + identnum))

# Finds all median images for a given date
# Returns a dictionary of median images, with keys being exposures.
def median_all_date(date, color=False):
    # I've hard coded the files for now, this can be changed later.
    directory = 'Images/Original/' + date + '/'

    # Gotta make sure those images exist.
    try:
        files = os.listdir(directory)
    except:
        print('Images directory not found for that date!')
        print('Are you sure you downloaded images?')
        exit()

    # These dictionaries hold the images and existence booleans.
    keys = ['All', 0.02, 0.3, 6]

    finalimg = {}
    superimg = {}
    exists = {}

    # By doing this with an array you can add more medians
    # just by adding them to the keys array.
    if not color:
        for key in keys:
            finalimg[key] = np.zeros((512, 512))
            superimg[key] = np.zeros((1, 1, 1))
            exists[key] = False
    else:
        for key in keys:
            finalimg[key] = np.zeros((512, 512, 3))
            superimg[key] = np.zeros((1, 1, 1, 1))

    # If not color load all the ones and seperate by exposure time
    # If color, then just load all of them ignoring exposure, for now.
    if not color:
        for file in files:
            # Make sure we look in the directory to load the image.
            file = directory + file

            # We have to reshape the images so that the lowest level
            # single value is a 1D array rather than just a number.
            # This is so when you concat the arrays it actually turns the
            # lowest value into a multivalue array.
            img = ndimage.imread(file, mode='L')
            temp = img.reshape(img.shape[0], img.shape[1], 1)

            exposure = ImageIO.get_exposure(img)

            # All Median
            # Make the super image have the correct
            # dimensions and starting values.
            # Concats if it already does.
            if exists['All']:
                # Concatenates along the color axis
                superimg['All'] = np.concatenate((superimg['All'], temp), axis=2)
            else:
                # Since we run this only once this shortcut will save us
                # fractions of a second!
                superimg['All'] = temp
                exists['All'] = True

            # Exposure specific medians
            if exists[exposure]:
                superimg[exposure] = np.concatenate((superimg[exposure], temp), axis=2)
            else:
                superimg[exposure] = temp
                exists[exposure] = True
    else:
        superimg['All'] = ImageIO.load_all_date(date)

    print("Loaded images")

    # Axis 2 is the color axis (In RGB space 3 is the color axis).
    # Axis 0 is y, axis 1 is x iirc.
    # Can you believe that this is basically the crux of this method?
    for key in keys:

        # If not color we can use magic np median techniques.
        if not color:
            finalimg[key] = np.median(superimg[key], axis=2)
        # In color we use the median of median because rgb tuples.
        else:
            #final = finalimg[key]
            # Let's run this loop as little as possible thanks.
            if not np.array_equal(superimg[key], np.zeros((1, 1, 1, 1))):
                supe = superimg[key]
                #final = np.zeros((supe.shape[0], supe.shape[1], 3))

                x = 0
                y = 0
                for row in supe:
                    for column in row:
                        tuples = ndarray_to_tuplelist(column)
                        median = median_of_medians(tuples, len(tuples) // 2)
                        finalimg[key][y,x] = [median[1], median[2], median[3]]
                        x += 1
                    y += 1
                    x = 0
            #finalimg[key] = np.zeros((supe.shape[0], supe.shape[1], 3))
    print('Median images complete for ' + date)
    return finalimg

# Date tells this function what folder to save the medians in.
# Color tells us if the medians are in color or not.
def save_medians(medians, date, color=False):
    if not color:
        loc = 'Images/Median/' + date + '/'
        cmap = 'gray'
    else:
        loc = 'Images/Median-Color/' + date + '/'
        cmap = None
    
    for key, median in medians.items():
        name = str(key).replace('.', '')
        
        # If blocks to only save the ones with actual data
        if not color and not np.array_equal(median, np.zeros((1, 1))):
            ImageIO.save_image(median, name, loc, cmap)
            
        elif color and not np.array_equal(median, np.zeros((512, 512, 3))):
            ImageIO.save_image(np.uint8(median), name, loc)

if __name__ == "__main__":
    date = '20160823'
    #ImageIO.download_all_date(date)
    medians = median_all_date(date)
    save_medians(medians, date)
