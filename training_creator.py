import os
import random
import time

from PIL import Image
from matplotlib.widgets import RectangleSelector
import numpy as np

# The following two lines force matplotlib to use TkAgg as the backend.
# Blitting seems to be broken for me on Qt5Agg which is why I've switched.
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button

import coordinates
import image
import mask
from io_util import DateHTMLParser
import io_util


class TaggableImage:
    def __init__(self, name, update=False):
        self.press = False
        self.name = name

        # Artists for blitting.
        self.artists = []

        # Loads the image and reshapes to 512 512 in greyscale.
        if not update:
            loc = os.path.join(os.path.dirname(__file__), *["Images", "data", "to_label", name])
        else:
            loc = os.path.join(os.path.dirname(__file__), *["Images", "data", "train", "0.3", name])
        with open(loc, 'rb') as f:
            img = Image.open(f).convert('L')
            self.img = np.asarray(img).reshape((512, 512))

        # The grid division
        self.div = 16
        self.good = True

        self.update = update

        # The masking image that we're creating.
        self.mask = np.zeros((512, 512), dtype="uint8")

        if update:
            loc = os.path.join(os.path.dirname(__file__), *["Images", "data", "labels", "0.3", name])
            with open(loc, 'rb') as f:
                self.mask = np.array(Image.open(f).convert('L'))


    def set_up_plot(self):
        # Sets up the figure and axis.
        fig, ax = plt.subplots()
        fig.set_size_inches(15,15)
        self.artists.append(ax.imshow(self.img, cmap='gray'))

        # Circle at 30 degrees altitude, where the training patches end.
        circ1 = Circle(coordinates.center, radius=167, fill=False, edgecolor="cyan")
        self.artists.append(ax.add_patch(circ1))

        # Extra ten pixels in the radius so we are sure to get any pixels that
        # would be caught in the training patches.
        circ2 = Circle(coordinates.center, radius=167+10, fill=False, edgecolor="green")
        self.artists.append(ax.add_patch(circ2))

        # This is a little hacky but it recreates the grid shape with individual
        # rectangular patches. This way the grid can be updated with the rest
        # of the image upon clicking.
        for i in range(1, 24):
            height = i * self.div
            r = Rectangle((64, 64), height, height, edgecolor="m", fill=False)
            self.artists.append(ax.add_patch(r))

            r = Rectangle((64, 448-height), height, height, edgecolor="m", fill=False)
            self.artists.append(ax.add_patch(r))

            r = Rectangle((448-height, 64), height, height, edgecolor="m", fill=False)
            self.artists.append(ax.add_patch(r))

            r = Rectangle((448-height, 448-height), height, height, edgecolor="m", fill=False)
            self.artists.append(ax.add_patch(r))

        # Shows the divisions on the x and y axis.
        grid = np.arange(0, 513, self.div)
        plt.xticks(grid)
        plt.yticks(grid)

        self.artists.append(ax.imshow(self.mask, cmap="viridis", alpha=0.25, vmin=0, vmax=255, animated=True))

        self.ax = ax
        self.fig = fig


    def on_click(self, event):
        self.press = True
        # Finds the box that the click was in.
        x = event.xdata
        x = int(x//self.div * self.div)

        y = event.ydata
        y = int(y//self.div * self.div)

        # Sets the box to white, and then updates the plot with the new
        # masking data.
        if not self.update:
            self.mask[y:y + self.div, x:x + self.div] = 255
        else:
            self.mask[y:y + self.div, x:x + self.div] = 128
        self.artists[-1].set_data(self.mask)

        # Does the blitting update
        for a in self.artists:
            self.ax.draw_artist(a)
        self.fig.canvas.blit(self.ax.bbox)


    def on_motion(self, event):
        # If we're not currently clicked while moving don't make the box white.
        if not self.press:
            return

        # Don't want to run all this code if the box we're in is already white.
        if self.mask[int(event.ydata), int(event.xdata)] == 255:
            return

        # Finds the box that we're currently in.
        x = event.xdata
        x = int(x//self.div * self.div)

        y = event.ydata
        y = int(y//self.div * self.div)

        # Sets the box to white, and then updates the plot with the new
        # masking data.
        if not self.update:
            self.mask[y:y + self.div, x:x + self.div] = 255
        else:
            self.mask[y:y + self.div, x:x + self.div] = 128
        self.artists[-1].set_data(self.mask)

        # Does the blitting update
        for a in self.artists:
            self.ax.draw_artist(a)
        self.fig.canvas.blit(self.ax.bbox)


    def on_release(self, event):
        self.press = False


    def connect(self):
        cidpress = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        cidmove = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)


    def save(self):
        # When the plot is closed we save the newly created label mask.
        save_im = image.AllSkyImage(self.name, None, None, self.mask)
        exp_im = image.AllSkyImage(self.name, None, None, self.img)
        exp = image.get_exposure(exp_im)
        loc = os.path.join(os.path.dirname(__file__), *["Images", "data", "labels", str(exp)])

        # Maks the antenna
        m = mask.generate_mask()
        save_im = mask.apply_mask(m, save_im)

        # Saves the image.
        image.save_image(save_im, loc)

        if not self.update:
            # Moves the downloaded image into the training folder.
            loc = os.path.join(os.path.dirname(__file__), *["Images", "data", "to_label", self.name])
            dest = os.path.join(os.path.dirname(__file__), *["Images", "data", "train", str(exp), self.name])
            os.rename(loc, dest)
            print("Moved: " + loc)

    def cleanup(self, event):
        # Deletes the downloaded image so that we don't have it clogging
        # everything up.
        loc = os.path.join(os.path.dirname(__file__), *["Images", "data", "to_label", self.name])
        os.remove(loc)
        self.good = False
        print("Deleted: " + loc)



def get_image(update, i=0):
    if not update:
        # The link to the camera.
        link = "http://kpasca-archives.tuc.noao.edu/"

        # This extracts the dates listed and then picks one at random.
        data = io_util.download_url(link)
        htmldata = data.text
        parser = DateHTMLParser()
        parser.feed(htmldata)
        parser.close()
        date = random.choice(parser.data)
        parser.clear_data()

        link = link + date

        # This extracts the images from the given date and then picks at random.
        data = io_util.download_url(link)
        htmldata = data.text
        parser.feed(htmldata)
        parser.close()
        image = random.choice(parser.data)

        # This loop ensures that we don't accidentally download the all night
        # gifs or an image in a blue filter.
        while image == 'allblue.gif' or image == 'allred.gif' or image[:1] == 'b':
            image = random.choice(parser.data)

        # Once we have an image name we download it to Images/data/to_label
        # First we need to make sure it exists.
        label_location = os.path.join(os.path.dirname(__file__), *["Images", "data", "to_label"])
        if not os.path.exists(label_location):
            os.makedirs(label_location)

        # Downloads the image
        io_util.download_image(date[:8], image, directory=label_location)

        # Returns the image name.
        return image
    else:
        images = os.listdir(os.path.join(os.path.dirname(__file__), *["Images", "data", "train", "0.3"]))
        images = sorted(images)
        if images[0] == ".DS_Store":
            return images[i+1]
        return images[i]



if __name__ == "__main__":
    done = {}
    update = False
    # The list of all the pictures that have already been finished.
    finished_location = os.path.join(os.path.dirname(__file__), *["Images", "data", "labels", "0.3"])
    if not os.path.exists(finished_location):
        os.makedirs(finished_location)
    done["0.3"] = os.listdir(finished_location)

    # Separate out the 0.3s and 6s images.
    finished_location = os.path.join(os.path.dirname(__file__), *["Images", "data", "labels", "6"])
    if not os.path.exists(finished_location):
        os.makedirs(finished_location)
    done["6"] = os.listdir(finished_location)

    i = 0
    # We run this loop until the user kills the program.
    while True:

        name = get_image(update, i)

        # Loads the image into the frame to label.
        if not update and ((not name in done["0.3"]) or (not name in done["6"])):
            im = TaggableImage(name)
            im.set_up_plot()
            im.connect()

            # Adds the bad image button
            b_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
            button = Button(b_ax, "Bad Image")
            button.on_clicked(im.cleanup)

            plt.show()
            print(im.good)
            if im.good:
                im.save()
                i += 1

            print("Num images:" + str(i))

        elif update:
            im = TaggableImage(name, update)
            im.set_up_plot()
            im.connect()

            # Adds the bad image button
            b_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
            button = Button(b_ax, "Bad Image")
            button.on_clicked(im.cleanup)

            plt.show()

            if im.good:
                im.save()

            i += 1
