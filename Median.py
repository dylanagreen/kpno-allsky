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
    #print('Success')

#download_all_date('20170711')





# Converts the PIL image to a numpy array image.
# i2 = np.array(i)


# Below this is the original difference file I'll prolly need later.

# Reads the two images into ndarrays.
# Importing in RGB mode makes it ignore the A layer. 
# If we suddenly have transparency I might have a problem.
#img1 = ndimage.imread('unnamed.png', mode = 'RGB')
#img2 = ndimage.imread('unnamed-2.png', mode = 'RGB')

# Difference image. Assumed literal mathematical difference for now.
# I encountered a problem previously, in that I assumed the type of the array would dynamically change.
# This is python, so that's not wrong per se. Anyway turns out it's wrong so I have to cast these to numpy ints.
# I then have to cast back to uints because imshow works differently on uint8 and int16.
#diffimg = np.uint8(abs(np.int16(img1) - np.int16(img2)))

#print(diffimg) # Debug line

# Generate Figure and Axes objects.
#figure = plot.figure()
#figure.set_size_inches(4,4) # 4 inches by 4 inches
#axes = plot.Axes(figure,[0.,0.,1.,1.]) # 0 - 100% size of figure


# Turn off the actual visual axes for visual niceness.
# Then add axes to figure
#axes.set_axis_off()
#figure.add_axes(axes)

# Adds the image into the axes and displays it
#axes.imshow(diffimg)


# DPI chosen to have resultant image be the same size as the originals. 128*4 = 512
#plot.savefig("blah.png", dpi = 128)

# Show the plot
#plot.show()

