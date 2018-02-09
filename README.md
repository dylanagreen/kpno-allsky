# kpno-allsky

This is some scripts/software I’m developing based on all-sky images from the Kitt Peak National Observatory. These images can be found at the following url: http://kpasca-archives.tuc.noao.edu

Current dependencies:
AstroPy
SciPy
NumPy
matplotlib
Requests
At present:

# ImageIO.py
- Finds the difference between two images.
- Saves images.
- Gets the exposure of images.
- Downloads all the images from a given day.

# Median.py
- Finds a median image for all images from a given day.

# Coordinates.py
- Converts the x,y pixel coordinates of an image to the ra,dec for that pixel.
- Converts ra,dec to x,y pixel coordinates.
- Can find a star near a given pixel
- Makes corrections to az/r coords to increase conversion accuracy.

# Mask.py
- Takes a few medians and finds the hot/stuck pixels.
- Outputs as an image or a list of pixels.
- Additional Masks mask the horizon objects, and all pixels outside the circular image.

# Transform.py
- Converts a circular all-sky image into an ra-dec image that shows the ra-dec visible portions of the sky for that image.
- Ra-dec image is projected as an Eckert-IV projection.
- Methods are provided to transform arrays of ra and dec into x,y for Mollweide and Eckert-IV projections.
- Ignores masked pixels from Mask.

# Clouds.py
- Makes (most of) the clouds in 6s exposure images darker.
- Does some .3s clouds well and others badly.

# Histogram.py
- Creates histograms of greyscale values in .3s images.
- Categorizes the histograms based on predefined categories.
- Finds an approximate radius of the moon in each image.

# Notebooks
This repo also features a few jupyter notebooks designed for quick plotting/model corrections. Their functions are defined at the top of the notebooks themselves.