from scipy import ndimage
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plot
import os
import math

import ImageIO

# Radii in kilometers
# Radius of the earth is the radius of the earth shadow at the moon's orbit.
R_moon = 1737
R_earth = 4479

# Returns the amount of the moon that is lit up still by the sun, and not
# shaded by the earth.
def eclipse_visible(d, R, r):

    # This makes addition work as addition and not concatenates
    d = np.asarray(d)

    # Part 1 of the shaded area equation
    a = (d * d + r * r - R * R)
    b = d * 2 * r
    p1 = r * r * np.arccos(a / b)

    # Part 2 of the shaded area equation
    a = (d * d + R * R - r * r)
    b = d * 2 * R
    p2 = R * R * np.arccos(a / b)

    # Part 3 of the shaded area equation
    a1 = (r + R - d)
    a2 = (d + r - R)
    a3 = (d - r + R)
    a4 = (d + r + R)
    p3 = (0.5) * np.sqrt(a1 * a2 * a3 * a4)

    # Add them together to get the shaded area
    A = p1 + p2 - p3

    # Get the shaded proportion by divding the shaded area by the total area
    # Assumes r is the radius of the moon being shaded.
    
    P = A / (np.pi * r * r)

    # P is the shaded area, so 1-P is the lit up area.
    P = 1 - P

    return P


# Finds the size of the moon region (approximately) by taking pixels that are
# "close to white" (in this case, > 255 - threshold)
def moon_size(date, file):
    img = ndimage.imread('Images/Original/' + date + '/' + file, mode = 'L')

    thresh = 5
    img = np.where(img >= 255 - thresh, 1, 0)

    # Runs a closing to smooth over local minimums (which are mainly caused by
    # a rogue antenna). Then labels the connected white regions. Structure s
    # Makes it so that regions connected diagonally are counted as 1 region.
    s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    img = ndimage.morphology.binary_closing(img, structure = s)
    labeled, nums = ndimage.label(img, structure = s)

    # Want to find the size of each labeled region.
    sizes = [0] * (nums + 1)
    for row in labeled:
        for val in row:
            sizes[val] = sizes[val] + 1

    # We want to exclude the background from the "biggest region" calculation.
    # It's quicker to just set the background (0) region to 0 than to check
    # every single value above before adding it to the array.
    sizes[0] = 0

    # Following code calculates d, the distance between the center of
    # Earth's shadow and the center of the moon. Basically just d = v*t.

    # 0 = Start of moon being shaded
    # 1 = totality of eclipse
    eclipse_0 = (10) * 3600 + (49.8) * 60
    eclipse_1 = (12) * 3600 + (51.4) * 60

    # Time of this image in seconds
    time = int(file[4:6]) * 3600 + int(file[6:8]) * 60 + int(file[8:10])

    # Point slope form.
    slope = (R_moon + R_earth) / (eclipse_0 - eclipse_1)
    d = slope * (time - eclipse_1)

    # The moon is the biggest white area... most of the time.
    # TODO: Change this using a find moon function to just pick the region that
    # astropy says the moon is in.
    biggest = np.amax(sizes)

    return (biggest, d)


if __name__ == "__main__":
    date = '20180131'

    directory = 'Images/Original/' + date + '/'
    files = sorted(os.listdir(directory))

    f = open('phase.txt', 'w')

    distances = []
    found = []

    for file in files:
        biggest, d = moon_size(date, file)

        found.append(biggest)
        distances.append(d)


    f.close()

    vis = eclipse_visible(distances, R_earth, R_moon)
    print(vis)

    yes = False
    i1 = 0
    i2 = 0
    for i in range(0,len(vis)):
        if not math.isnan(vis[i]) and not yes:
            yes = True
            i1 = i
        if math.isnan(vis[i]) and yes:
            yes = False
            i2 = i
            break

    print(i1)
    print(i2)

    plot.scatter(vis[i1: i2 + 1], found[i1: i2 + 1])
    plot.ylabel("Approx Moon Size (pixels)")
    plot.xlabel("Proportion of moon visible")

    plot.show()
    