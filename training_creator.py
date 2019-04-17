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

import coordinates
import image


class TaggableImage:
    def __init__(self, name):
        self.press = False
        self.name = name

        # Artists for blitting.
        self.artists = []

        # Loads the image and reshapes to 512 512 in greyscale.
        loc = os.path.join('Images', *['data', 'cloud', name])
        with open(loc, 'rb') as f:
            img = Image.open(f).convert('L')
            self.img = np.asarray(img).reshape((512, 512))

        # The grid division
        self.div = 16


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

        # The masking image that we're creating.
        self.mask = np.zeros((512, 512), dtype="uint8")
        self.artists.append(ax.imshow(self.mask, cmap="gray", alpha=0.25, vmin=0, vmax=255, animated=True))

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
        self.mask[y:y+self.div, x:x+self.div] = 255
        self.artists[-1].set_data(self.mask)

        # Does the blitting update
        for a in self.artists:
            self.ax.draw_artist(a)
        self.fig.canvas.blit(self.ax.bbox)


    def on_motion(self, event):
        # If we're not currently clicked while moving don't make the box white.
        if not self.press:
            return

        # Don't want to run all this code if the box we're in is already
        # white.
        if self.mask[int(event.ydata), int(event.xdata)] == 255:
            return

        # Finds the box that we're currently in.
        x = event.xdata
        x = int(x//self.div * self.div)

        y = event.ydata
        y = int(y//self.div * self.div)

        # Sets the box to white, and then updates the plot with the new
        # masking data.
        self.mask[y:y+self.div, x:x+self.div] = 255
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
        loc = os.path.join('Images', *['data', 'labels-2', '0.3'])
        image.save_image(save_im, loc)

# Gets all the possible pictures to label and then shuffles the order.
pics = os.listdir(os.path.join('Images', *['data', 'cloud']))
random.shuffle(pics)

# The list of all the pictures that have already been finished.
done = os.listdir(os.path.join('Images', *['data', 'labels-2', '0.3']))

# Loop through all the pictures.
for name in pics:
    if not name in done:
        im = TaggableImage(name)
        im.set_up_plot()
        im.connect()

        plt.show()
        im.save()