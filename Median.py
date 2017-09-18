import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import requests
import os
from html.parser import HTMLParser
from scipy import ndimage
from PIL import Image
from io import BytesIO

import ImageIO

# Html parser for looping through html tags
class DateHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.data = []

    def handle_starttag(self, tag, attrs):
        # All image names are held in tags of form <A HREF=imagename>
        if tag == 'a':
            for attr in attrs:
                # If the first attribute is href we need to ignore it
                if attr[0] == 'href':
                    self.data.append(attr[1])


def download_all_date(date):
    # Creates the link
    link = 'http://kpasca-archives.tuc.noao.edu/' + date

    directory = 'Images/' + date
    # Verifies that an Images folder exists, creates one if it does not.
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Gets the html for a date page,
    # then parses it to find the image names on that page.
    htmllink = link + '/index.html'
    rdate = requests.get(htmllink)
    htmldate = rdate.text
    parser = DateHTMLParser()
    parser.feed(htmldate)
    parser.close()
    imagenames = parser.data

    # Runs through the array of image names and downloads them
    for image in imagenames:
        # We want to ignore the all image animations
        if image == 'allblue.gif' or image == 'allred.gif':
            continue
        # Otherwise request the html data of the page for that image
        # and save the image
        else:
            imageloc = link + '/' + image
            imagename = directory + '/' + image
            rimage = requests.get(imageloc)

            # Converts the image data to a python image
            i = Image.open(BytesIO(rimage.content)).convert('RGB')
            # Saves the image
            i.save(imagename)

    print('All photos downloaded for ' + date)


# Gets the exposure time of an image.
# 225 is the greyscale value for the yellow in the image.
# The pixel chosen is yellow for .03s and "not yellow" for 6s.
# The second pixel is yellow in .03 and .002
# but due to magic of if blocks that's ok.
def get_exposure(image):
    if image[19][174] == 225:
        return '0.3'
    if image[17][119] == 225:
        return '0.02'
    else:
        return '6'


# Loads all the images for a certain date
def load_all_date(date):

    # I've hard coded the files for now, this can be changed later.
    directory = 'Images/' + date + '/'

    # In theory this is only ever called from median_all_date.
    # Just in case though.
    try:
        files = os.listdir(directory)
    except:
        print('Images directory not found for that date!')
        print('Are you sure you downloaded images?')
        exit()

    dic = {}
    imgs = len(files)
    n = 0

    # Runs while the number of 100 blocks doesn't encompass all the images yet.
    while n <= (imgs // 100):
        if (n+1)*100 < imgs:
            final = 100
        else:
            final = (imgs - n*100)

        if final == 0:
            break
        else:
            file = directory + files[n * 100]
            temp = gray_and_color_image(file)

        # Creates the array of images.
        for i in range(1, final):
            # Loads in the image and creates that imgtemp
            file = directory + files[i + n * 100]

            imgtemp = gray_and_color_image(file)

            # i + n * 100 required for > 100 images
            temp = np.concatenate((temp, imgtemp), axis=3)

        n += 1
        if final > 0:
            dic[n] = temp

    # Return is the super image for later.
    # This just makes it random and in the correct shape for later
    # In case key == 1 fails.
    result = np.random.rand(512, 512, 4, 1)

    for key, val in dic.items():
        # result doesn't exist yet so set it to val for the first key.
        if key == 1:
            result = val
        else:
            result = np.concatenate((result, val), axis=3)

    return result


# Loads in an image and returns it as an array where
# each pixel has 4 values associated with it:
# Grayscale (L), R, G and B
def gray_and_color_image(file):
    img = ndimage.imread(file, mode='RGB')
    img2 = ndimage.imread(file, mode='L')

    # Reshape to concat
    img2 = img2.reshape(img2.shape[0], img2.shape[1], 1)
    img = np.concatenate((img2, img), axis=2)

    # Return the reshaped image
    return img.reshape(img.shape[0], img.shape[1], 4, 1)


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
    directory = 'Images/' + date + '/'

    # Gotta make sure those images exist.
    try:
        files = os.listdir(directory)
    except:
        print('Images directory not found for that date!')
        print('Are you sure you downloaded images?')
        exit()

    # These dictionaries hold the images and existence booleans.
    keys = ['All', '0.02', '0.3', '6']

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

            exposure = get_exposure(img)

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
        superimg['All'] = load_all_date(date)

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
    
    if not os.path.exists(loc):
        os.makedirs(loc)
    
    for key, median in medians.items():
        name = key.replace('.', '')
        
        # If blocks to only save the ones with actual data
        if not color and not np.array_equal(median, np.zeros((1, 1))):
            ImageIO.save_image(median, name, loc, cmap)
            
        elif color and not np.array_equal(median, np.zeros((512, 512, 3))):
            ImageIO.save_image(np.uint8(median), name, loc)

date = '20170918'
#download_all_date(date)
medians = median_all_date(date)
save_medians(medians, date)
