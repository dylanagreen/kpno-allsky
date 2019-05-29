"""A module providing facilities for creating masking arrays for all-sky images.

This module contains methods to create three different kinds of masking arrays.
The masks are as follows:

- One that masks hot pixels
- One that masks hot pixels and any horizon objects
- One that masks hot pixels, horizon objects and everything outside the
  circular image.

Further methods are provided to save a given mask as an image,
and to apply a masking array to an image.
"""

import os
import math
import numpy as np
from PIL import Image

import image
from image import AllSkyImage


def generate_clean_mask():
    """Generate a clean mask for KPNO images.

    Generates a masking array for KPNO images that only masks out hot pixels.

    Returns
    -------
    numpy.ndarray
        The mask array where 1 represents pixels that are to be masked and 0
        represents pixels that should remain visible.

    Notes
    -----
    generate_clean_mask requires there to be median images in Images/Mask/.
    These images can be downloaded from the kpno-allsky github or may be
    generated by median.median_all_date and moved.

    """
    fileloc = os.path.join(os.path.dirname(__file__), *["Images", "Mask"])
    files = os.listdir(fileloc)

    tolerance = 160

    # Sets up the mask to be the right size
    file1 = os.path.join(fileloc, files[0])
    img = np.asarray(Image.open(file1).convert("L"))
    mask = np.zeros((img.shape[0], img.shape[1]))

    for f in files:
        f = os.path.join(fileloc, f)
        img = np.asarray(Image.open(f).convert("L"))

        for y in range(0, img.shape[1]):
            for x in range(0, img.shape[0]):
                # 255 is pure white so accept pixels between
                # white-tolerance and white
                # Y is first value as it"s the row value
                if img[y, x] >= (255 - tolerance):
                    mask[y, x] += 1

    # Get only the pixels that appear as "hot" in ALL of the images.
    # Set those to 0 to mask them.
    final = np.where(mask >= np.amax(mask), 1, 0)

    return final


def generate_mask(forcenew=False):
    """Generate a mask for KPNO images.

    Generates a masking array for KPNO images that masks out not only hot
    pixels, but also the horizon objects.

    Parameters
    ----------
    forcenew : bool, optional
        Whether or not this method should load a previously saved mask or if it
        should generate it completely from scratch.

    Returns
    -------
    numpy.ndarray
        The mask array where 1 represents pixels that are to be masked and 0
        represents pixels that should remain visible.

    See Also
    --------
    generate_clean_mask : Used by generate_mask to generate the hot pixel
                          mask.

    Notes
    -----
    generate_mask requires there to be median images in Images/Mask/ but also
    additionally requires an image named Ignore.png in Images/ that
    deliniates the horizon objects to be ignored.
    These images can be downloaded from the kpno-allsky github or may be
    generated by median.median_all_date and moved.

    """
    center = (256, 252)

    # Read in the ignore image.
    # I read this in first to make sure the Mask.png is the correct dimensions.
    ignore_loc = os.path.join(os.path.dirname(__file__), *["Images", "Ignore.png"])
    ignore = np.asarray(Image.open(ignore_loc).convert("RGB"))

    # If we"ve already generated and saved a mask, load that one.
    # This speeds up code execution by a lot, otherwise we loop through 512x512
    # pixels 6 times! With this we don"t have to even do it once, we just load
    # and go.
    maskloc = os.path.join(os.path.dirname(__file__), *["Images", "Mask.png"])
    if os.path.isfile(maskloc) and not forcenew:
        mask = np.asarray(Image.open(maskloc).convert("L"))
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

    # If we"ve made a new mask, save it so we can skip the above steps later.
    save_mask(mask)

    return mask


def generate_full_mask(forcenew=False):
    """Generates a complete mask for KPNO images.

    Generates a masking array for KPNO images that masks out not only hot
    pixels, but also the horizon objects and then additionally masks pixels
    outside of the circular all-sky image.

    Parameters
    ----------
    forcenew : bool, optional
        Whether or not this method should load a previously saved mask or if it
        should generate it completely from scratch.

    Returns
    -------
    numpy.ndarray
        The mask array where 0 represents pixels that are to be masked and 1
        represents pixels that should remain visible.

    See Also
    --------
    generate_mask : Used by generate_full_mask to generate the hot pixel
                    and horizon mask.

    Notes
    -----
    generate_full_mask calls generate_mask, which requires there to be
    median images in Images/Mask/ but also additionally requires an image
    named Ignore.png in Images/ that deliniates the horizon objects to be
    ignored. These images can be downloaded from the kpno-allsky github.
    """
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


def save_mask(mask):
    """Save a masking image.

    Parameters
    ----------
    mask : numpy.ndarray
        The mask to save.

    See Also
    --------
    image.save_image : Save an image.

    """
    img = AllSkyImage("Mask.png", None, None, mask * 255)
    image.save_image(img, "Images/")


def apply_mask(mask, img):
    """Apply a mask to a given image.

    Parameters
    ----------
    mask : numpy.ndarray
        The mask to apply.
    img : image.AllSkyImage
        The image to apply the mask to.

    Returns
    -------
    image.AllSkyImage
        The masked image, where masked pixels have been set to 0.

    Notes
    -----
    In order to apply a mask, it is inverted and then multiplied by the image.
    This zeroes out the masked pixels while leaving the non masked pixels
    untouched. In a rigorous analysis of the image, the masked pixels will
    count as zeroes. In these cases it is better to use the generated mask
    in conjunction with NumPy"s masked arrays to complete ignore the pixels
    you want to mask, instead of setting them to 0.

    """
    data = np.copy(img.data)

    # Multiply mask with image to keep only non masked pixels.
    # Invert the mask first since masked pixels are 1 and non masked are 0,
    # And we want the non mask to be 1 so they get kept.
    mask = 1 - mask
    data = np.multiply(mask, data)

    new = AllSkyImage(img.name, img.date, img.camera, data)

    return new

if __name__ == "__main__":
    generate_mask(True)
