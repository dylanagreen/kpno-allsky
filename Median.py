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
    
    # Create the final image array first.
    finalimg = np.zeros((512,512))

    # Boolean for if we have our temp image array yet.
    superimg = np.zeros((1,1,1))
    exists = False
    for file in files:
        # Make sure we look in the directory lol.
        file = dir + file
    
        # We have to reshape the images so that the lowest level single value is a 1D array rather than just a number.
        # This is so when you concat the arrays it actually turns the lowest value into a multivalue array.
        img = ndimage.imread(file, mode = 'L')
        temp = img.reshape(512, 512, 1)
    
        # Make the super image have the correct dimensions and starting values.
        if exists:
            # Concatenates along the color axis
            superimg = np.concatenate((superimg,temp), axis=2)
        else:
            # Since we run this only once this shortcut will save us fractions of a second!
            superimg = temp
            exists = True
        


    # i and j are counters here for positions in the final array.
    # i = row
    # j = column
    i = 0
    j = 0

    for row in superimg:
        for column in row:
            # Need to sort before taking the median.
            # Median is just the middle item in the array so find the length, then integer divide by 2.
            # This keeps it an integer but still rounds up if the length was odd. (i.e. median in 5 items is item 3 => index 2 => 5 // 2)
            sorted = np.sort(column)
            pos = sorted.shape[0] // 2 
            finalimg[i,j] = sorted[pos]
            #print(sorted[pos])
            j += 1
        i += 1
        j = 0



    # Generate Figure and Axes objects.
    figure = plot.figure()
    figure.set_size_inches(4,4) # 4 inches by 4 inches
    axes = plot.Axes(figure,[0.,0.,1.,1.]) # 0 - 100% size of figure


    # Turn off the actual visual axes for visual niceness.
    # Then add axes to figure
    axes.set_axis_off()
    figure.add_axes(axes)

    # Adds the image into the axes and displays it
    # cmap is required here since I did the images in grayscale and imshow needs to know that.
    axes.imshow(finalimg, cmap = 'gray')

    filename = 'Images/' + date + '-median.png'
    # DPI chosen to have resultant image be the same size as the originals. 128*4 = 512
    plot.savefig(filename, dpi = 128)
    
    print('Median image complete for ' + date)
    
    # Show the plot
    plot.show()
    


date = '20170710'
download_all_date(date)
median_all_date(date)




# Converts the PIL image to a numpy array image.
# i2 = np.array(i)

