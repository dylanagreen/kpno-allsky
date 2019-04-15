import os
import random

from PIL import Image
from matplotlib.widgets import RectangleSelector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import coordinates
import image

pics = os.listdir(os.path.join('Images', *['data', 'cloud']))
random.shuffle(pics)
done = os.listdir(os.path.join('Images', *['data', 'labels-2', '0.3']))
for name in pics:
    if not name in done:
        loc = os.path.join('Images', *['data', 'cloud', name])
        with open(loc, 'rb') as f:
            im = Image.open(f).convert('L')
            im = np.asarray(im).reshape((512, 512))

        fig, ax = plt.subplots()
        fig.set_size_inches(15,15)
        ax.imshow(im, cmap='gray')

        # Extra ten pixels in the radius so we are sure to get any pixels that
        # would be caught in the training patches.
        circ1 = Circle(coordinates.center, radius=167+10, fill=False, edgecolor='green')

        # Circle of training patches just to see
        circ2 = Circle(coordinates.center, radius=167, fill=False, edgecolor='cyan')
        ax.add_patch(circ1)
        ax.add_patch(circ2)

        div = 8

        x = np.arange(0, 513, div)
        plt.xticks(x)
        plt.yticks(x)
        plt.grid(color='m')

        mask = np.zeros((512, 512), dtype='uint8')
        m_obj = plt.imshow(mask, cmap='gray', alpha=0.25, vmin=0, vmax=255)

        def onclick(event):
            # Finds the box that the click was in.
            x = event.xdata
            x = int(x//div * div)

            y = event.ydata
            y = int(y//div * div)

            # Sets the box to white, and then updates the plot with the new
            # masking data.
            mask[y:y+div, x:x+div] = 255
            m_obj.set_data(mask)
            plt.draw()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        test = image.AllSkyImage(name, None, None, mask)
        loc = os.path.join('Images', *['data', 'labels-2', '0.3'])
        image.save_image(test, loc)
