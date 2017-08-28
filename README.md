# kpno-allsky

This is some scripts/software I’m developing based on all-sky images from the kitt-peak observatory. It should probably have version control so I set this up.

Current dependencies:
AstroPy
SciPy
NumPy
mayplotlib
Requests

At present:

# Difference.py
- Finds the difference between two images.

# Median.py
- Downloads all the images from a given day
- Finds a median image for all images from a given day.

# Coordinates.py
- Converts the x,y pixel coordinates of an image to the ra,dec for that pixel.
- Converts ra,dec to x,y pixel coordinates.
- Can find a star near a given pixel
- Makes corrections to az/r coords to increase conversion accuracy.

# Mask.py
- Takes a few medians and finds the hot/stuck pixels.
- Outputs as an image or a list of pixels.

# Convert.py
- Converts a circular all-sky image into an ra-dec image that shows the ra-dec visible portions of the sky for that image.
- Ignores horizon objects, as defined in Ignore.png

# Notebooks
This repo also features a few jupyter notebooks designed for quick plotting/model corrections. Their functions are defined at the top of the notebooks themselves.