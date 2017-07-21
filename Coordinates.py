import math
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.time.core as aptime
from astropy import units as u
import numpy as np

from matplotlib.patches import Circle
import matplotlib.image as image
import matplotlib.pyplot as plot
from scipy import ndimage

# Returns a tuple of form (alt, az)
def xy_to_altaz(x,y):
    # Center of the circle found using super acccurate photoshop layering techniques.
    center = (256, 252)

    # Point adjusted based on the center being at... well... the center. And not the top left. In case you were confused.
    # Y is measured from the top stop messing it up.
    pointadjust = (x - center[0], center[1] - y)

    az = math.atan2((pointadjust[0]),(pointadjust[1]))

    # atan2 ranges from -pi to pi so we have to shift the -pi to 0 parts up 2 pi.
    if pointadjust[0] < 0:
        az += 2 * math.pi


    az = math.degrees(az)


    # Pythagorean thereom boys.
    r = math.sqrt(pointadjust[0]**2 + pointadjust[1]**2)

    # For now if r is on the edge of the circle or beyond we'll have it just be 0 degrees. (Up from horizontal)
    if r >= 239:
        alt = 0
    else:
        #r = 238 # Debug line
        r = r * 11.6 / 239 # Magic pixel to mm conversion rate

        rpoints = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 11.6]
        thetapoints = [0, 3.58, 7.17, 10.76, 14.36, 17.98, 21.62, 25.27, 28.95, 32.66, 36.40, 40.17, 43.98, 47.83, 51.73, 55.67, 59.67, 63.72, 67.84, 72.03, 76.31, 80.69, 85.21, 89.97,90]

        # 90- turns the angle from measured from the vertical to measured from the horizontal.
        # This interpolates the value from the two on either side of it.
        alt = 90 - np.interp(r, xp = rpoints, fp = thetapoints)

    return (alt,az)

# Returns a tuple of form (ra, dec)
def altaz_to_radec(alt, az, time):
    assert type(time) is aptime.Time, "Time should be an astropy Time Object."

    # This is the latitutde/longitude of the camera
    cameraloc = (31.959417, -111.598583)

    cameraearth = EarthLocation(lat = cameraloc[0] * u.deg, lon = cameraloc[1] * u.deg, height = 2120 * u.m)

    altazcoord = SkyCoord(alt = alt * u.deg, az = az * u.deg, frame = 'altaz', obstime = time, location = cameraearth)
    radeccoord = altazcoord.icrs

    return radeccoord

# Returns a tuple of form (alt, az)
def radec_to_altaz(ra, dec, time):
    assert type(time) is aptime.Time, "Time should be an astropy Time Object."

    # This is the latitutde/longitude of the camera
    cameraloc = (31.959417, -111.598583)

    cameraearth = EarthLocation(lat = cameraloc[0] * u.deg, lon = cameraloc[1] * u.deg, height = 2120 * u.meter)

        
    # Creates the SkyCoord object
    radeccoord = SkyCoord(ra = ra, dec = dec, unit = 'deg', obstime = time, location = cameraearth, frame = 'icrs', temperature = 5 * u.deg_C, pressure = 78318 * u.Pa)

    # Handy
    #polaris = SkyCoord.from_name('polaris', 'icrs')

    # Transforms
    altazcoord = radeccoord.transform_to('altaz')

    return (altazcoord.alt.degree, altazcoord.az.degree)

# Returns a tuple of form (x,y)
def altaz_to_xy(alt, az):
    rpoints = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 11.6]
    thetapoints = [0, 3.58, 7.17, 10.76, 14.36, 17.98, 21.62, 25.27, 28.95, 32.66, 36.40, 40.17, 43.98, 47.83, 51.73, 55.67, 59.67, 63.72, 67.84, 72.03, 76.31, 80.69, 85.21, 89.97,90]
    
    r = np.interp(90 - alt, xp = thetapoints, fp = rpoints)

    r = r * 239 / 11.6 # mm to pixel rate

    # Remember angle measured from vertical so sin and cos are swapped from usual polar.
    # These are x,y's with respect to a zero.
    x = -1 * r * math.sin(math.radians(az))
    y = r * math.cos(math.radians(az))
    
    
    # Center of the circle found using super acccurate photoshop layering techniques.
    center = (256, 252)

    # y is measured from the top!
    pointadjust = (x + center[0], center[1] - y)
    return pointadjust

# Returns a tuple of form (x,y)
def radec_to_xy(ra, dec, time):
    altaz = radec_to_altaz(ra, dec, time)
    return altaz_to_xy(altaz[0], altaz[1])
    
# Time must be an astropy Time object.
# Please.
def timestring_to_obj(date, filename):
    # Add the dashes
    formatted = date[:4] + '-' + date[4:6] + '-' + date[6:]

    # Extracts the time from the file name. File names seem to be machine generated so this should not break.
    # Hopefully.
    time = filename[4:6] + ':' + filename[6:8] + ':' + filename[8:10]

    formatted = formatted + ' ' + time

    return Time(formatted)


def celestialhorizon(date, file):
    time = timestring_to_obj(date, file)
    
    file = 'Images/' + date + '/' + file + '.png'
    img = ndimage.imread(file, mode = 'RGB')
    

    
    dec = 0
    ra = 0
    while ra <= 360:
        xy = radec_to_xy(ra, dec, time)
        xy = (round(xy[0]), round(xy[1]))
               
        # Remember y is first, then x
        # Also make sure it's on the image at all.
        if xy[1] < 512 and xy[0] < 512:
            img[xy[1], xy[0]] = (244, 66, 229)
        
        ra += 0.5

    return img
    
    
# Draws a circle at the x y coord with radius 5.
# I've appropriated this as a save image method for now.
def circle(x,y,img):

    x = round(x)
    y = round(y)
    
    # Generate Figure and Axes objects.
    figure = plot.figure()
    figure.set_size_inches(4,4) # 4 inches by 4 inches
    axes = plot.Axes(figure,[0.,0.,1.,1.]) # 0 - 100% size of figure

    # Turn off the actual visual axes for visual niceness.
    # Then add axes to figure
    axes.set_axis_off()
    figure.add_axes(axes)

    # Adds the image into the axes and displays it
    axes.imshow(img)
    
    axes.set_aspect('equal')
    circ = Circle((x, y), 5, fill = False)
    circ.set_edgecolor('c')
    axes.add_patch(circ)

    # DPI chosen to have resultant image be the same size as the originals. 128*4 = 512
    plot.savefig("blah.png", dpi = 128)

    # Show the plot
    #plot.show()


altaz = xy_to_altaz(250,300)


#print('Azimuthal angle = ' + str(altaz[1]))
#print('Altitude angle = ' + str(altaz[0]))


tempfile = 'r_ut043526s01920'
date = '20170721'

#print(altaz_to_radec(altaz[0],altaz[1],timestring_to_obj('20170719', tempfile)))

#star = SkyCoord.from_name('Polaris', 'icrs')
#print(star)

point = radec_to_xy(37.9461429,  89.2641378, timestring_to_obj(date, tempfile))

#print(point)

img = celestialhorizon(date, tempfile)
circle(point[0],point[1],img)

