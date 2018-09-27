import os
import math
import time
from html.parser import HTMLParser
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import requests
from requests.exceptions import (TooManyRedirects, HTTPError, ConnectionError,
                                 Timeout, RequestException)


# Reads a link, with exception handling and error checking built in.
# Returns a requests.Response object if it succeeds, returns None if it fails.
def download_url(link):
    """Read the data at a url.
    
    Parameters
    ----------
    link : str
        The link to access and download data from.
    
    Returns
    -------
    requests.Response or None
        If the link was successfully read and downloaded, returns a requests
        Response object. If not, returns None and raises an exception.
    
    Raises
    ------
    TooManyRedirects
        If there are too many redirects in the link provided.
    HTTPError
        If there is a generic HTTP error associated with reading the link.
    ConnectionError
        If the script was unable to create a connection to the link.
    Timeout
        If the link does not provide any information after three attempts.
    RequestException
        If Requests raises an exception that is not covered by the previous
        four exceptions.
    
    """
    tries = 0
    read = False

    while not read:
        try:
            # Tries to connect for 5 seconds.
            data = requests.get(link, timeout=5)

            # Raises the HTTP error if it occurs.
            data.raise_for_status()
            read = True

        # Too many redirects is when the link redirects you too much.
        except TooManyRedirects:
            print('Too many redirects.')
            return None
        # HTTPError is an error in the http code.
        except HTTPError:
            print('HTTP error with status code ' + str(data.status_code))
            return None
        # This is a failure in the connection unrelated to a timeout.
        except ConnectionError:
            print('Failed to establish a connection to the link.')
            return None
        # Timeouts are either server side (too long to respond) or client side
        # (when requests doesn't get a response before the timeout timer is up)
        # I have set the timeout to 5 seconds
        except Timeout:
            tries += 1

            if tries >= 3:
                print('Timed out after three attempts.')
                return None

            # Tries again after 5 seconds.
            time.sleep(5)

        # Covers every other possible exceptions.
        except RequestException as err:
            print('Unable to read link')
            print(err)
            return None
        else:
            print(link + ' read with no errors.')
            return data


# Saves an input image with the given name in the folder denoted by location.
# If the image is greyscale, cmap should be 'gray'
def save_image(img, name, location, cmap=None, patch=None):
    """Save an image.
    
    Save an image passed in `img` with the name `name` into the location in
    `location`. `cmap` provides an option to save the image in greyscale, and
    `patch` allows matplotlib patches to be applied on top of the saved image.
    
    Parameters
    ----------
    img : ndarray
        The image to be saved, as type ``ndarray``.
    name : str
        The name of the saved image.
    location : str
        The relative path to save the image to. If the path does not exist,
        it is created.
    cmap : str, optional
        A colormap to use when saving the image. For grayscale images, use
        'gray,' otherwise defaults to no colormap.
    patch : matplotlib.patches.Patch, optional
        A matplotlib patch to apply on top of the saved image. By default no 
        patch is applied.
    
    """
    if not os.path.exists(location):
        os.makedirs(location)

    # DPI chosen to have resultant image be the same size as the originals.
    # 128*4 = 512
    dpi = 128
    y = img.shape[0] / dpi
    x = img.shape[1] / dpi

    # Generate Figure and Axes objects.
    figure = plt.figure()
    figure.set_size_inches(x, y)  # 4 inches by 4 inches
    axes = plt.Axes(figure, [0., 0., 1., 1.])  # 0 - 100% size of figure

    # Turn off the actual visual axes for visual niceness.
    # Then add axes to figure
    axes.set_axis_off()
    figure.add_axes(axes)

    # Adds the image into the axes and displays it
    # Then saves
    axes.imshow(img, cmap=cmap)

    if patch:
        axes.add_patch(patch)

    # If location was passed with / on the end, don't append another one.
    if not location[-1:] == '/':
        name = location + '/' + name
    else:
        name = location + name

    # Append .png if it wasn't passed in like that already.
    if not name[-4:] == '.png':
        name = name + '.png'

    # Print "saved" after saving, in case saving messes up.
    plt.savefig(name, dpi=dpi)
    print('Saved: ' + name)

    # Close the plot in case you're running multiple saves.
    plt.close()


# Html parser for looping through html tags
class DateHTMLParser(HTMLParser):
    """Parser for data passed from image websites.
    
    Attributes
    ----------
    data : list
        Extracted data from the image website HTML.
    """
    def __init__(self):
        HTMLParser.__init__(self)
        self.data = []

    def handle_starttag(self, tag, attrs):
        """Extract image links from the HTML start tag.
        
        Parameters
        ----------
        tag : str
            The start tag
        attrs : list
            The attributes attached to the corresponding `tag`.
        
        """
        # All image names are held in tags of form <A HREF=imagename>
        if tag == 'a':
            for attr in attrs:
                # If the first attribute is href we need to ignore it
                if attr[0] == 'href':
                    self.data.append(attr[1])

    def clear_data(self):
        """Clear the data list of this parser instance.
        
        """
        self.data = []


# Downloads all the images for a certain date for a given camera.
# Currently supports the kpno all sky and the mmto all sky.
def download_all_date(date, camera="kpno"):
    """Download all images for a given date and all-sky camera.
    
    Parameters
    ----------
    date : str
        Date to download images for, in the form yyyymmdd.
    camera : str, optional
        Camera to download images from. Defaults to `kpno` (the all-sky camera
        at Kitt-Peak) but may be specified instead as `mmto` (the all-sky
        camera at the MMT Observatory).
    
    Raises
    ------
    TooManyRedirects
        If there are too many redirects in downloading the images.
    HTTPError
        If there is a generic HTTP error associated with downlading images.
    ConnectionError
        If the script was unable to create a connection to the image link.
    Timeout
        If the image link does not provide any information after three attempts.
    RequestException
        If Requests raises an exception that is not covered by the previous
        four exceptions.
    
    Notes
    -----
    Over the course of the run time of this method various status updates will
    be printed. The method will exit early with a print out of what happened
    and a raised exception if the image link is unable to be read. 
    
    The Kitt-Peak National Observatory images are located at 
    http://kpasca-archives.tuc.noao.edu/.
    
    The MMT Observatory images are located at 
    http://skycam.mmto.arizona.edu/skycam/.
    """
    links = {'kpno': 'http://kpasca-archives.tuc.noao.edu/',
             'mmto': 'http://skycam.mmto.arizona.edu/skycam/'}

    # Creates the link
    link = links[camera] + date

    # Gets the html for a date page,
    # then parses it to find the image names on that page.
    if camera == 'kpno':
        htmllink = link + '/index.html'
    else:
        htmllink = link

    rdate = download_url(htmllink)

    if rdate is None:
        print('Failed to download dates.')
        return

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
    imagenames2 = []
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
            download_image(date, image, camera)

    print('All photos downloaded for ' + date)


def download_image(date, image, camera='kpno'):
    """Download a single image.
    
    This method is of a similar form to download_all_date, where `date` 
    provides the date and `camera` provides the camera. `image` is the name
    of the image to be downloaded. Images are saved to 
    
    Parameters
    ----------
    date : str
        Date to download images for, in the form yyyymmdd.
    image : str
        Image name to download.
    camera : str, optional
        Camera to download images from. Defaults to `kpno` (the all-sky camera
        at Kitt-Peak) but may be specified instead as `mmto` (the all-sky
        camera at the MMT Observatory).
    
    Raises
    ------
    TooManyRedirects
        If there are too many redirects in downloading the images
    HTTPError
        If there is a generic HTTP error associated with downlading the image.
    ConnectionError
        If the script was unable to create a connection to the image link.
    Timeout
        If the image link does not provide any information after three attempts.
    RequestException
        If Requests raises an exception that is not covered by the previous
        four exceptions.
    
    Notes
    -----
    Over the course of the run time of this method various status updates will
    be printed. The method will exit early and fail to downlod the image 
    with a failure print out and a raised exception if the image link is 
    unable to be read. 
    
    The Kitt-Peak National Observatory images are located at 
    http://kpasca-archives.tuc.noao.edu/.
    
    The MMT Observatory images are located at 
    http://skycam.mmto.arizona.edu/skycam/.
    """
    links = {'kpno': 'http://kpasca-archives.tuc.noao.edu/',
             'mmto': 'http://skycam.mmto.arizona.edu/skycam/'}

    # Creates the link
    link = links[camera] + date

    # Collects originals in their own folder within Images
    directory = 'Images/Original/' + camera.upper() + '/' + date
    # Verifies that an Images folder exists, creates one if it does not.
    if not os.path.exists(directory):
        os.makedirs(directory)

    imageloc = link + '/' + image
    imagename = directory + '/' + image

    rimage = download_url(imageloc)

    if rimage is None:
        print('Failed: ' + imagename)
        return

    # Saves the image
    with open(imagename, 'wb') as f:
        f.write(rimage.content)
    print("Downloaded: " + imagename)


def load_all_date(date):
    """Load all images for a given date.
    
    Parameters
    ----------
    date : str
        The date in formate yyyymmdd. 
    
    Returns
    -------
    ndarray
        An ``ndarray`` that contains all images for that date. ``ndarray`` is 
        of the shape (512, 512, 4, N) where N is the number of images for 
        that day.
    
    See Also
    --------
    gray_and_color_image : Images are loaded using gray_and_color_image.
    
    """

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

    # This makes the result random and in the correct shap in case key == 1
    # fails.
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
    """Load an image in both grayscale and color.
    
    Load an image and return an image where each pixel is represented by a 
    four item list, of the form [L, R, G, B] where L is the luma grayscale
    value.
    
    Parameters
    ----------
    file : str
        The location of the image to be read in.
    
    Returns
    -------
    ndarray
        The ndarray representing the grayscale and color combination image.
    
    Notes
    -----
    
    The SciPy documentation includes the following definition of the 
    ITU-R 601-2 luma grayscale transform:
    
        L = R * 299/1000 + G * 587/1000 + B * 114/1000
    
    """
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
    """ Get the exposure time of an image.
    
    Parameters
    ----------
    image : ndarray
        An ``ndarray`` representing the image data.
    
    Returns
    -------
    float or int
        The exposure time in seconds of the provided image. 
        Possible values are 0.3, 0.02 or 6.
    
    Notes
    -----
    get_exposure works by looking at two specific pixels in an image taken on
    the KPNO camera. The first pixel is at (174, 19) in (x, y) coordinates, 
    where (0, 0) is the top left corner of the image. This pixel appears as 
    gray in images taken at 0.3s or 0.02s exposure times, but as 
    black in images taken in 6s exposure times. In order to differentiate
    between 0.3s and 0.02s a second pixel at (119, 17) is used, which appears
    as gray in images taken at 0.02s exposure time but as black in images taken
    in 0.3s exposure time. 
    
    """
    pix1 = image[19, 174]
    pix2 = image[17, 119]

    # Handles separate cases for greyscale and RGB images.
    # Greyscale conversion below is the same one used by imread.
    if len(image.shape) == 3:
        pix1 = pix1[0] * 299/1000 + pix1[1] * 587/1000 + pix1[2] * 114/1000
        pix1 = math.floor(pix1)

        pix2 = pix2[0] * 299/1000 + pix2[1] * 587/1000 + pix2[2] * 114/1000
        pix2 = math.floor(pix2)

    if pix1 == 225:
        return 0.3
    if pix2 == 225:
        return 0.02
    return 6


# Returns the difference image between two images.
# Black areas are exactly the same in both, white areas are opposite.
# Greyscale/color values are varying levels of difference.
def image_diff(img1, img2):
    """Find the mathematical difference between two grayscale images.
    
    Parameters
    ----------
    img1 : ndarray
        The first image.
    img2 : ndarray
        The second image.
    
    Notes
    -----
    The order of the parameters does not matter. In essence, 
    image_diff(img1, img2) == image_diff(img2, img1).
    """
    # I encountered a problem previously, in that
    # I assumed the type of the array would dynamically change.
    # This is python, so that's not wrong per se.
    # Anyway turns out it's wrong so I have to cast these to numpy ints.
    # I then have to cast back to uints because imshow
    # works differently on uint8 and int16.
    diffimg = np.uint8(abs(np.int16(img1) - np.int16(img2)))

    return diffimg


if __name__ == "__main__":
    for i in range(20180101, 20180132):
        date = str(i)
        download_all_date(date)
