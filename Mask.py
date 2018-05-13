from scipy import ndimage
import numpy as np
import math
import os

import ImageIO


# Looks through the median images to make the mask and returns a mask array.
# "Clean" in this case means no horizon objects.
# The mask thus only contains likely "hot" pixels.
def generate_clean_mask():
    fileloc = 'Images/Mask/'
    files = os.listdir(fileloc)

    tolerance = 160

    # Sets up the mask to be the right size
    file1 = fileloc + files[0]
    img = ndimage.imread(file1, mode='L')
    mask = np.zeros((img.shape[0], img.shape[1]))

    for file in files:
        file = fileloc + file
        img = ndimage.imread(file, mode='L')

        for y in range(0, img.shape[1]):
            for x in range(0, img.shape[0]):
                # 255 is pure white so accept pixels between
                # white-tolerance and white
                # Y is first value as it's the row value
                if img[y, x] >= (255 - tolerance):
                    mask[y, x] += 1

    # Get only the pixels that appear as "hot" in ALL of the images.
    # Set those to 0 to mask them.
    final = np.where(mask >= np.amax(mask), 1, 0)

    return final


# Gets the "clean" mask and then adds the horizon objects to it.
# forcenew allows you to force this to disregarded a saved mask.
# Pixels that should be masked are returned with a '1' value.
# Pixels that should be unchanged are returned with a '0' value.
def generate_mask(forcenew=False):
    center = (256, 252)

    # Read in the ignore image.
    # I read this in first to make sure the Mask.png is the correct dimensions.
    ignore = ndimage.imread('Images/Ignore.png', mode='RGB')

    # If we've already generated and saved a mask, load that one.
    # This speeds up code execution by a lot, otherwise we loop through 512x512
    # pixels 6 times! With this we don't have to even do it once, we just load
    # and go.
    maskloc = 'Images/Mask.png'
    if os.path.isfile(maskloc) and not forcenew:
        mask = ndimage.imread(maskloc, mode='L')
        # Converts the 255 bit loaded image to binary 1-0 image.
        mask = np.where(mask == 255, 1, 0)

        # Have to compare these two separately, since ignore has a third color
        # dimensions and mask.shape == ignore.shape would therefore always be
        # False.
        if mask.shape[0] == ignore.shape[0] and mask.shape[1] == ignore.shape[1]:
            return mask

    # Get the "clean" mask, i.e. the pixels only ignore mask.
    mask = generate_clean_mask()

    hyp = math.hypot
    for y in range(0, ignore.shape[1]):
        for x in range(0, ignore.shape[0]):
            x1 = x - center[0]
            y1 = center[1] - y
            r = hyp(x1, y1)

            # Ignore horizon objects (which have been painted pink)
            # Only want the horizon objects actually in the circle.
            # Avoids unnecessary pixels.
            if r < 242 and np.array_equal(ignore[y, x], [244, 66, 235]):
                mask[y, x] = 1


    # If we've made a new mask, save it so we can skip the above steps later.
    save_mask(mask)

    return mask


# The full mask is the horion objects plus the hot pixels, plus then it blacks
# out everything that's not in the circular image.
def generate_full_mask(forcenew=False):
    mask = generate_mask(forcenew)
    center = (256, 252)

    hyp = math.hypot
    # Ignore everything outside the circular image.
    for y in range(0, mask.shape[1]):
        for x in range(0, mask.shape[0]):
            x1 = x - center[0]
            y1 = center[1] - y
            r = hyp(x1, y1)
            if r > 241:
                mask[y, x] = 1

    return mask


# Saves a given mask as in image in the Images folder.
def save_mask(mask):
    ImageIO.save_image(mask, 'Mask.png', 'Images/', 'gray')


# Applies the mask to the image
def apply_mask(mask, img):
    img = np.copy(img)

    # Multiply mask with image to keep only non masked pixels.
    # Invert the mask first since masked pixels are 1 and non masked are 0,
    # And we want the non mask to be 1 so they get kept.
    mask = 1 - mask
    img = np.multiply(mask, img)

    return img
