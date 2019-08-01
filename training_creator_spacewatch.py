#!/usr/bin/env python3
import os
import random
from shutil import copyfile
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
            img = Image.open(f).convert("RGB")
            self.img = np.asarray(img).reshape((1024, 1024, 3))

        # The grid division
        self.div = 16
        self.good = True

        self.update = update

        # The masking image that we're creating.
        self.mask = np.zeros((1024, 1024), dtype="uint8")

        if update:
            loc = os.path.join(os.path.dirname(__file__), *["Images", "data", "labels", "0.3", name])
            with open(loc, 'rb') as f:
                self.mask = np.array(Image.open(f).convert('L'))

        # This is gray, but we change it to 255 if we're labeling ghosts.
        self.val = 128


    def set_up_plot(self):
        # Sets up the figure and axis.
        fig, ax = plt.subplots()
        fig.set_size_inches(15,15)
        self.artists.append(ax.imshow(self.img, cmap='gray'))

        # Circle at 30 degrees altitude, where the training patches end.
        circ1 = Circle(coordinates.center_sw, radius=330, fill=False, edgecolor="cyan")
        self.artists.append(ax.add_patch(circ1))

        # Extra ten pixels in the radius so we are sure to get any pixels that
        # would be caught in the training patches.
        circ2 = Circle(coordinates.center_sw, radius=330+15, fill=False, edgecolor="green")
        self.artists.append(ax.add_patch(circ2))

        # This is a little hacky but it recreates the grid shape with individual
        # rectangular patches. This way the grid can be updated with the rest
        # of the image upon clicking.
        for i in range(1, 45):
            height = i * self.div
            r = Rectangle((160, 160), height, height, edgecolor='m', fill=False)
            self.artists.append(ax.add_patch(r))

            r = Rectangle((160, 864 - height), height, height, edgecolor='m', fill=False)
            self.artists.append(ax.add_patch(r))

            r = Rectangle((864 - height, 160), height, height, edgecolor='m', fill=False)
            self.artists.append(ax.add_patch(r))

            r = Rectangle((864 - height, 864 - height), height, height, edgecolor='m', fill=False)
            self.artists.append(ax.add_patch(r))

        # Shows the divisions on the x and y axis.
        grid = np.arange(0, 1023, self.div * 2)
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
            self.mask[y:y + self.div, x:x + self.div] = self.val
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
        if self.mask[int(event.ydata), int(event.xdata)] == self.val:
            return

        # Finds the box that we're currently in.
        x = event.xdata
        x = int(x//self.div * self.div)

        y = event.ydata
        y = int(y//self.div * self.div)

        # Sets the box to white, and then updates the plot with the new
        # masking data.
        if not self.update:
            self.mask[y:y + self.div, x:x + self.div] = self.val
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
        save_im = image.AllSkyImage(self.name, None, None, self.mask)

        loc = os.path.join(os.path.dirname(__file__), *["Images", "data", "labels-sw"])

        # Saves the image.
        image.save_image(save_im, loc)

        if not self.update:
            # Moves the downloaded image into the training folder.
            loc = os.path.join(os.path.dirname(__file__), *["Images", "data", "to_label", self.name])
            dest = os.path.join(os.path.dirname(__file__), *["Images", "data", "train-sw", self.name])
            os.rename(loc, dest)
            print("Moved: " + loc)

    def cleanup(self, event):
        # Deletes the downloaded image so that we don't have it clogging
        # everything up.
        loc = os.path.join(os.path.dirname(__file__), *["Images", "data", "to_label", self.name])
        os.remove(loc)
        self.good = False
        print("Deleted: " + loc)


    # Swaps the tag value from cloud to ghost.
    def swap(self, event):
        if self.val == 255:
            self.val = 128
        else:
            self.val = 255


def get_image(update, i=0):
    if not update:
        base_loc = os.path.join(os.path.dirname(__file__), *["Images", "Original", "SW"])
        all_dates = os.listdir(base_loc)
        date = random.choice(all_dates)

        all_images = os.listdir(os.path.join(base_loc, date))
        image = random.choice(all_images)

        # Once we have an image name we copy it to Images/data/to_label
        # First we need to make sure it exists.
        label_loc = os.path.join(os.path.dirname(__file__), *["Images", "data", "to_label"])
        if not os.path.exists(label_loc):
            os.makedirs(label_loc)

        # Downloads the image
        copyfile(os.path.join(base_loc, *[date, image]), os.path.join(label_loc, image))

        # Returns the image name.
        return image
    else:
        images = os.listdir(os.path.join(os.path.dirname(__file__), *["Images", "data", "train", "0.3"]))
        images = sorted(images)
        if images[0] == ".DS_Store":
            return images[i+1]
        return images[i]



if __name__ == "__main__":
    update = False
    # The list of all the pictures that have already been finished.
    finished_loc = os.path.join(os.path.dirname(__file__), *["Images", "data", "labels-sw"])
    if not os.path.exists(finished_loc):
        os.makedirs(finished_loc)
    done = os.listdir(finished_loc)


    i = 0
    # We run this loop until the user kills the program.
    while True:

        name = get_image(update, i)

        # Loads the image into the frame to label.
        if not update and not name in done:
            im = TaggableImage(name)
            im.set_up_plot()
            im.connect()

            # Adds the swap button.
            b_ax = plt.axes([0.5, 0.1, 0.1, 0.05])
            button1 = Button(b_ax, "Swap Label")
            button1.on_clicked(im.swap)

            # Adds the bad image button
            b_ax = plt.axes([0.6, 0.1, 0.1, 0.05])
            button2 = Button(b_ax, "Bad Image")
            button2.on_clicked(im.cleanup)

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
