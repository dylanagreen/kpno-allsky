#!/usr/bin/env python3
"""A module providing facilities for manipulating spacewatch images.

"""

import datetime
import logging
import os
import time
from html.parser import HTMLParser

import ephem
import numpy as np
import pytesseract
import requests
from PIL import Image
from requests.exceptions import (TooManyRedirects, HTTPError, ConnectionError,
                                 Timeout, RequestException)


# Sets up a pyephem object for the camera.
# Using the lat/long of the other KPNO camera for now.
camera = ephem.Observer()
camera.lat = '31.959417'
camera.lon = '-111.598583'
camera.elevation = 2120


def download_url(link):
    """Read the data at a url.

    Parameters
    ----------
    link : str
        The link to access and download data from.

    Returns
    -------
    requests.Response or None
        A requests.Response object containing data on success,
        or None on failure.

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
            logging.error('Too many redirects.')
            return None
        # HTTPError is an error in the http code.
        except HTTPError:
            logging.error('HTTP error with status code ' + str(data.status_code))
            return None
        # This is a failure in the connection unrelated to a timeout.
        except ConnectionError:
            logging.error('Failed to establish a connection to the link.')
            return None
        # Timeouts are either server side (too long to respond) or client side
        # (when requests doesn't get a response before the timeout timer is up)
        # I have set the timeout to 5 seconds
        except Timeout:
            tries += 1

            if tries >= 3:
                logging.error('Timed out after three attempts.')
                return None

            # Tries again after 5 seconds.
            time.sleep(5)

        # Covers every other possible exceptions.
        except RequestException as err:
            logging.error('Unable to read link')
            print(err)
            return None
        else:
            logging.info(link + ' read with no errors.')
            return data


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
        # All image names are held in tags of form <img=imagename>
        if tag == 'img':
            for attr in attrs:
                # If the first attribute is href we need to ignore it
                if attr[0] == 'src':
                    self.data.append(attr[1])

    def clear_data(self):
        """Clear the data list of this parser instance.
        """
        self.data = []


def download_image(date):
    # Creates the link
    link = 'http://varuna.kpno.noao.edu/allsky/AllSkyCurrentImage.JPG'

    # Collects originals in their own folder within Images
    time = datetime.datetime.now()

    directory = os.path.join('Images', *['Original', 'SW', date])
    # This directory is for use on the blackbox server.
    #directory = os.path.join('/media', *['data1', 'spacewatch', date])

    # Verifies that an Images folder exists, creates one if it does not.
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Label the images in a similar way to the kpno ones.
    imagename = os.path.join(directory, "temp.png")

    rimage = download_url(link)

    if rimage is None:
        logging.error('Failed: ' + time.strftime('%Y%m%d %H:%M:%S'))
        return

    # Saves the image
    with open(imagename, 'wb') as f:
        f.write(rimage.content)

    # This extracts the text relevant portion of the image and resizes it so
    # that py tesseract can extract the text better.
    temp_im = Image.open(imagename).crop((120, 0, 240, 30)).resize((480, 120))
    text = pytesseract.image_to_string(temp_im)
    # Renames the image.
    new_name = 'c_ut' + text.replace(":", "") + '.png'
    os.rename(imagename, os.path.join(directory, new_name))
    logging.debug("Downloaded: " + new_name)


def run_and_download():
    while True:
        sun = ephem.Sun()
        # Sleep for an extra four minutes after sunrise to update rising.
        camera.date = datetime.datetime.utcnow()
        setting = camera.next_setting(sun, use_center=True).datetime()
        rising = camera.next_rising(sun, use_center=True).datetime()
        now = datetime.datetime.utcnow()

        # If the next rising is before the next setting then we're in the night
        #if not rising < setting:
         #   print("Current time:", now)
          #  print("Setting at:", setting)
           # print("Rising at:", rising)
            #delta = (setting - now).total_seconds()
            # Sleeps until the sun sets.
            #time.sleep(delta)
        print("Sunset arrived, starting download.")
        day = datetime.datetime.now().strftime('%Y%m%d')
        directory = os.path.join('Images', *['Original', 'SW', day])

        # This directory is for use on the blackbox server.
        #directory = os.path.join('/media', *['data1', 'spacewatch', day])

        # Verifies that an Images folder exists, creates one if it does not.
        if not os.path.exists(directory):
            os.makedirs(directory)

        log_name = os.path.join(directory, "download.log")
        logging.basicConfig(filename=log_name, level=logging.DEBUG)

        now = datetime.datetime.utcnow()
        while now < rising:
            try:
                download_image(day)

                # Sleeps until 6 seconds after two minutes from now.
                sleep_until = (datetime.datetime.now() + datetime.timedelta(seconds=120)).replace(second=6)
                sleep_for = (sleep_until - datetime.datetime.now()).total_seconds()
                time.sleep(sleep_for)

                now = datetime.datetime.utcnow()
            except Exception as e:
                logging.error(e)
                raise(e)


if __name__ == "__main__":
    run_and_download()

