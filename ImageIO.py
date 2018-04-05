import numpy as np
import matplotlib.pyplot as plot
import requests
import os
import math
from html.parser import HTMLParser
from PIL import Image
from io import BytesIO
from scipy import ndimage
from astropy.io import fits
from astropy.utils.data import download_file


# Saves an input image with the given name in the folder denoted by location.
# If the image is greyscale, cmap should be 'gray'
def save_image(img, name, location, cmap=None):
    if not os.path.exists(location):
        os.makedirs(location)

    # DPI chosen to have resultant image be the same size as the originals.
    # 128*4 = 512
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
    axes.imshow(img, cmap=cmap)

    # If location was passed with / on the end, don't append another one.
    if not location[-1:] == '/':
        name = location + '/' + name
    else:
        name = location + name

    # Append .png if it wasn't passed in like that already.
    if not name[-4:] == '.png':
        name = name + '.png'

    # Print "saved" after saving, in case saving messes up.
    plot.savefig(name, dpi=dpi)
    print('Saved: ' + name)

    # Close the plot in case you're running multiple saves.
    plot.close()


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


# Downloads all the images for a certain date for a given camera.
# Currently supports the kpno all sky and the mmto all sky.
def download_all_date(date, camera="kpno"):
    
    links = {'kpno' : 'http://kpasca-archives.tuc.noao.edu/', 
             'mmto' : 'http://skycam.mmto.arizona.edu/skycam/'}
    
    # Creates the link
    link = links[camera] + date

    # Prevents clutter by collecting originals in their own folder within Images
    directory = 'Images/Original/' + camera.upper() + '/' + date
    # Verifies that an Images folder exists, creates one if it does not.
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Gets the html for a date page,
    # then parses it to find the image names on that page.
    if camera == 'kpno':
        htmllink = link + '/index.html'
    else:
        htmllink = link

    rdate = requests.get(htmllink)
    
    # Makes sure the date exists.
    if rdate.status_code == 404:
        print("Date not found.")
        return
    
    htmldate = rdate.text
    parser = DateHTMLParser()
    parser.feed(htmldate)
    parser.close()
    imagenames = parser.data

    # Strips everything that's not a fits image.
    if camera == 'mmto':
        for item in imagenames:
            if item[-4:] == 'fits':
                imagenames2.append(item)
        imagenames = imagenames2
    
    # Runs through the array of image names and downloads them
    for image in imagenames:
        # We want to ignore the all image animations
        if image == 'allblue.gif' or image == 'allred.gif':
            continue
        # Otherwise request the html data of the page for that image
        # and save the image
        else:
            # I could use my ImageIO save_image here, but this way is quicker
            # since I don't have to make a plot for every image.
            imageloc = link + '/' + image
            imagename = directory + '/' + image
            rimage = requests.get(imageloc)

            # Saves the image
            with open(imagename, 'wb') as f:
                    f.write(rimage.content)
            print("Downloaded: " + imagename)

    print('All photos downloaded for ' + date)


# Loads all the images for a certain date
def load_all_date(date):

    # I've hard coded the files for now, this can be changed later.
    directory = 'Images/Original/' + date + '/'

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


# Gets the exposure time of an image.
# 225 is the greyscale value for the yellow in the image.
# The pixel chosen is yellow for .03s and "not yellow" for 6s.
# The second pixel is yellow in .03 and .002
# but due to magic of if blocks that's ok.
def get_exposure(image):

    # Handles separate cases for greyscale and RGB images.
    if len(image.shape) == 2:
        pix1 = image[19, 174]
        pix2 = image[17, 119]
    # Greyscale conversion below is the same one used by imread.
    elif len(image.shape) == 3:
        pix1 = image[19, 174]
        pix1 = pix1[0] * 299/1000 + pix1[1] * 587/1000 + pix1[2] * 114/1000
        pix1 = math.floor(pix1)

        pix2 = image[17, 119]
        pix2 = pix2[0] * 299/1000 + pix2[1] * 587/1000 + pix2[2] * 114/1000
        pix2 = math.floor(pix2)

    if pix1 == 225:
        return 0.3
    if pix2 == 225:
        return 0.02
    else:
        return 6


# Returns the difference image between two images.
# Black areas are exactly the same in both, white areas are opposite.
# Greyscale/color values are varying levels of difference.
def image_diff(img1, img2):
    # I encountered a problem previously, in that
    # I assumed the type of the array would dynamically change.
    # This is python, so that's not wrong per se.
    # Anyway turns out it's wrong so I have to cast these to numpy ints.
    # I then have to cast back to uints because imshow
    # works differently on uint8 and int16.
    diffimg = np.uint8(abs(np.int16(img1) - np.int16(img2)))

    return diffimg


if __name__ == "__main__":
    date = '20150404'
    download_all_date(date)