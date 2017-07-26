from scipy import ndimage
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import requests

from PIL import Image
from io import BytesIO
from html.parser import HTMLParser
import os

# Html parser for looping through html tags
class DateHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.data = []

    def handle_starttag(self, tag, attrs):
        # All image names are held in tags of form <A HREF=imagename>
        if tag == 'a':
            for attr in attrs:
                #If the first attribute is href we need to ignore it
                    if attr[0] == 'href':
                        #print("attr:", attr[1])
                        self.data.append(attr[1])

def download_all_date(date):
    # Creates the link
    link = 'http://kpasca-archives.tuc.noao.edu/' + date

    directory = 'Images/' + date
    # Verifies that an Images folder exists, creates one if it does not.
    if not os.path.exists(directory):
        os.makedirs(directory)


    # Gets the html for a date page, then parses it to find the image names on that page.
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
        if image == 'allblue.gif' or image ==  'allred.gif':
            continue
        # Otherwise request the html data of the page for that image and save the image
        else:
            imageloc = link + '/' + image
            imagename = directory + '/' + image
            rimage = requests.get(imageloc)

            # Converts the image data to a python image
            i = Image.open(BytesIO(rimage.content)).convert('RGB')
            # Saves the image
            i.save(imagename)

            # While testing I don't want to save a billion images so here's a print line
            #print('yes')

    print('All photos downloaded for ' + date)

# Gets the expousre time of an image. 225 is the greyscale value for the yellow in the image.
# The pixel chosen is yellow for .03s and "not yellow" for 6s.
# The second pixel is yellow in .03 and .002 but due to magic of if blocks that's ok.
def get_exposure(image):
    if image[19][174] == 225:
        return '0.3'
    if image[17][119] == 225:
        return '0.02'
    else:
        return '6'


def median_all_date(date):

    # I've hard coded the files for now, this can be changed later.
    directory = 'Images/' + date + '/'

    # Gotta make sure those images exist.
    try:
        files = os.listdir(directory)
    except:
        print('Images directory not found for that date!')
        print('Are you sure you downloaded images?')
        exit()

    # Directory for the files to save to later
    filedir = 'Images/Median/' + date

    if not os.path.exists(filedir):
        os.makedirs(filedir)

    # These dictionaries hold the images and existence booleans.
    keys = ['All', '0.02', '0.3', '6']

    finalimg = {}
    superimg = {}
    exists = {}

    # By doing this with an array you can add more medains just by adding them to the array.
    for key in keys:
        finalimg[key] = np.zeros((512,512))
        superimg[key] = np.zeros((1,1,1))
        exists[key] = False

    for file in files:
        # Make sure we look in the directory to load the image lol.
        file = directory + file

        # We have to reshape the images so that the lowest level single value is a 1D array rather than just a number.
        # This is so when you concat the arrays it actually turns the lowest value into a multivalue array.
        img = ndimage.imread(file, mode = 'L')
        temp = img.reshape(img.shape[0], img.shape[1], 1)

        exposure = get_exposure(img)

        # All Median
        # Make the super image have the correct dimensions and starting values. Concats if it already does.
        if exists['All']:
            # Concatenates along the color axis
            superimg['All'] = np.concatenate((superimg['All'],temp), axis=2)
        else:
            # Since we run this only once this shortcut will save us fractions of a second!
            superimg['All'] = temp
            exists['All'] = True

        # Exposure specific medians
        if exists[exposure]:
           superimg[exposure] = np.concatenate((superimg[exposure],temp), axis=2)
        else:
           superimg[exposure] = temp
           exists[exposure] = True

    print('Loaded images')
    # Sets the size of the image x,y in inches to be the same as the original
    dpi = 128
    y = superimg['All'].shape[0] / dpi
    x = superimg['All'].shape[1] / dpi

    # Generate Figure and Axes objects.
    figure = plot.figure()
    figure.set_size_inches(x,y) # x inches by y inches
    axes = plot.Axes(figure,[0.,0.,1.,1.]) # 0 - 100% size of figure


    # Turn off the actual visual axes for visual niceness.
    # Then add axes to figure
    axes.set_axis_off()
    figure.add_axes(axes)


    # Axis 2 is the color axis. Axis 0 is y, axis 1 is x iirc.
    # Can you believe that this is basically the crux of this method?
    # 100 lines of code to set up and save. 3 lines that actually make the images.
    for key in keys:
        finalimg[key] = np.median(superimg[key], axis = 2)

        # For brevity
        final = finalimg[key]

        # cmap is required here since I did the images in grayscale and imshow needs to know that.
        axes.imshow(final, cmap = 'gray')

        # Saves the pic
        key = key.replace('.', '')
        filename = filedir + '/' + key

        # I'm tired of saving blank black images lol.
        if not np.array_equal(final,np.zeros((1,1))):
            plot.savefig(filename, dpi = dpi)


    print('Median images complete for ' + date)

    # Show the plot
    #plot.show()



date = '1'
#download_all_date(date)
median_all_date(date)

#get_exposure(date)




# Converts the PIL image to a numpy array image.
# i2 = np.array(i)

