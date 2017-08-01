# kitt-peak

This is some scripts/software I’m developing based on all-sky images from the kitt-peak observatory. It should probably have version control so I set this up.

Current dependencies:
AstroPy
SciPy
NumPy
mayplotlib
Requests

At present:

#Difference.py
- Finds the difference between two images.

#Median.py
- Downloads all the images from a given day
- Finds a median image for all images from a given day.

#Coordinates.py
- Converts the x,y pixel coordinates of an image to the ra,dec for that pixel.
- Converts ra,dec to x,y pixel coordinates.