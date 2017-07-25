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

# Loads all the images for a certain date
def load_all_date(date):
    
    # I've hard coded the files for now, this can be changed later.
    directory = 'Images/' + date + '/'

    # In theory this is only ever called from median_all_date where this is already done.
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
        
        file = directory + files[n * 100]
        temp = gray_and_color_image(file)
        
        # Creates the array of images.
        for i in range(1,final):
            # Loads in the image and creates that imgtemp
            file = directory + files[i + n * 100]

            imgtemp = gray_and_color_image(file)
            
            # i + n * 100 required for > 100 images
            temp = np.concatenate((temp,imgtemp), axis = 3)
        
        n += 1
        if final > 0:
            dic[n] = temp
    
    # Return is the super image for later.
    # This just makes it random and in the correct shape for later, in case key == 1 fails.
    toreturn = np.random.rand(512,512,4,1)
    
    for key, val in dic.items():
        # toreturn doesn't exist yet so set it to val for the first key.
        if key == 1:
            toreturn = val
        else:
            toreturn = np.concatenate((toreturn,val), axis = 3)
    
    return toreturn

# Loads in an image and returns it as an array where each pixel has 4 values associated with it:
# Grayscale (L), R, G and B
def gray_and_color_image(file):
    img = ndimage.imread(file, mode = 'RGB')
    img2 = ndimage.imread(file, mode = 'L')

    # Reshape to concat
    img2 = img2.reshape(img2.shape[0], img2.shape[1], 1)
    img = np.concatenate((img2,img), axis = 2)

    # Return the reshaped image
    return img.reshape(img.shape[0], img.shape[1], 4, 1)

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
    filedir = 'Images/Median-Color/' + date

    if not os.path.exists(filedir):
        os.makedirs(filedir)

    # These dictionaries hold the images and existence booleans.
    keys = ['All', '0.02', '0.3', '6']

    finalimg = {}
    superimg = {}

    # By doing this with an array you can add more medains just by adding them to the keys array.
    for key in keys:
        finalimg[key] = np.zeros((512,512,3))
        superimg[key] = np.zeros((1,1,1,1))

    superimg['All'] = load_all_date(date)

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

    print("Loaded images")
    # Axis 2 is the color axis (In RGB space 3 is the color axis). Axis 0 is y, axis 1 is x iirc.
    # Can you believe that this is basically the crux of this method?
    for key in keys:
        # For brevity
        final = finalimg[key]
        
        if not np.array_equal(superimg[key],np.zeros((1,1,1,1))): # Let's run this loop as little as possible thanks.
            supe = superimg[key]
            final = np.zeros((supe.shape[0], supe.shape[1], 3))
            #supe = supe.reshape(supe.shape[0],supe.shape[1],supe.shape[-1],4) # This doesn't work as expected.
 
            x = 0
            y = 0
            for row in supe:
                for column in row:
                    #print(column)
                    tuples = ndarray_to_tuplelist(column)
                    median = median_of_medians(tuples,len(tuples) // 2)
                    #print(median) # Debug line
                    final[y][x] = [median[1], median[2], median[3]]
                
                    x +=1
                y += 1
                x = 0

        #np.set_printoptions(threshold=np.nan)
        #print(final)
        # Saves the pic
        key = key.replace('.', '')
        filename = filedir + '/' + key
        if not np.array_equal(final,np.zeros((512,512,3))):
            axes.imshow(np.uint8(final))
            plot.savefig(filename, dpi = dpi)

    print('Median images complete for ' + date)

    # Show the plot
    #plot.show()

# This is necessary.
def ndarray_to_tuplelist(arr):
    templist = []
    
    # Runs over the second dimension (the longer one lol)
    for i in range(0, arr.shape[1]):
        tup = (arr[0,i], arr[1,i], arr[2,i], arr[3,i])
        #print(tup) #Debug line
        templist.append(tup)
    
    return templist

# This works as wanted for tuples yay!
def median_of_medians(arr, i):

    # Divide the array into sublists of length 5 and find the medians.
    sublists = []
    medians = []

    for j in range (0, len(arr), 5):
        temp = arr[j:j+5]
        sublists.append(temp)
     
    for sublist in sublists:
        medians.append(sorted(sublist)[len(sublist)//2])

    if len(medians) <= 5:
        pivot = sorted(medians)[len(medians)//2]
    else:
        pivot = median_of_medians(medians,len(medians)//2) # Find the median of the medians array using this method lol.
        
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

date = '1'
#download_all_date(date)
median_all_date(date)

#get_exposure(date)


# Converts the PIL image to a numpy array image.
# i2 = np.array(i)

