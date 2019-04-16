import os
import random
import time

from PIL import Image
from matplotlib.widgets import RectangleSelector
import numpy as np

# The following two lines force matplotlib to use TkAgg as the backend.
# Blitting seems to be broken for me on Qt5Agg which is why I've made the
# switch.
import matplotlib
matplotlib.use('TkAgg')
#print(matplotlib.rcParams['backend'])
#import matplotlib.rcsetup as rcsetup
#print(rcsetup.all_backends)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import coordinates
import image

# Gets all the possible pictures to label and then shuffles the order.
pics = os.listdir(os.path.join('Images', *['data', 'cloud']))
random.shuffle(pics)

# The list of all the pictures that have already been finished.
done = os.listdir(os.path.join('Images', *['data', 'labels-2', '0.3']))

# Loop through all the pictures.
for name in pics:
    if not name in done:
        # Artists for blitting.
        artists = []
        # Loads the image and reshames to 512 512 in greyscale.
        loc = os.path.join('Images', *['data', 'cloud', name])
        with open(loc, 'rb') as f:
            img = Image.open(f).convert('L')
            img = np.asarray(img).reshape((512, 512))

        # Sets up the figure and axis.
        fig, ax = plt.subplots()
        fig.set_size_inches(15,15)
        artists.append(ax.imshow(img, cmap='gray'))

        # Circle at 30 degrees altitude, where the training patches end.
        circ1 = Circle(coordinates.center, radius=167, fill=False, edgecolor="cyan")
        artists.append(ax.add_patch(circ1))

        # Extra ten pixels in the radius so we are sure to get any pixels that
        # would be caught in the training patches.
        circ2 = Circle(coordinates.center, radius=167+10, fill=False, edgecolor="green")
        artists.append(ax.add_patch(circ2))

        # This is a little hacky but it recreates the grid shape with individual
        # rectangular patches. This way the grid can be updated with the rest
        # of the image upon clicking.
        for i in range(1, 24):
            height = i * 16
            r = Rectangle((64, 64), height, height, edgecolor="m", fill=False)
            artists.append(ax.add_patch(r))

            r = Rectangle((64, 448-height), height, height, edgecolor="m", fill=False)
            artists.append(ax.add_patch(r))

            r = Rectangle((448-height, 64), height, height, edgecolor="m", fill=False)
            artists.append(ax.add_patch(r))

            r = Rectangle((448-height, 448-height), height, height, edgecolor="m", fill=False)
            artists.append(ax.add_patch(r))

        # The grid division
        div = 16

        # Plots the magenta grid over the top to see where each block ends.
        grid = np.arange(0, 513, div)
        plt.xticks(grid)
        plt.yticks(grid)
        #plt.grid(True, color='m', zorder=10)

        # The masking image that we're creating.
        mask = np.zeros((512, 512), dtype="uint8")
        artists.append(ax.imshow(mask, cmap="gray", alpha=0.25, vmin=0, vmax=255, animated=True))

        def onclick(event):
            time1 = time.time()
            # Finds the box that the click was in.
            x = event.xdata
            x = int(x//div * div)

            y = event.ydata
            y = int(y//div * div)

            # Sets the box to white, and then updates the plot with the new
            # masking data.
            mask[y:y+div, x:x+div] = 255
            artists[-1].set_data(mask)

            # Does the blitting update
            for a in artists:
                ax.draw_artist(a)
            fig.canvas.blit(ax.bbox)

            time2 = time.time()
            print("Time: " + str(time2-time1))

        # Links the click to the click function and then shows the plot.
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        # When the plot is closed we save the newly created label mask.
        test = image.AllSkyImage(name, None, None, mask)
        loc = os.path.join('Images', *['data', 'labels-2', '0.3'])
        image.save_image(test, loc)
