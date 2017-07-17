import math
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from astropy import units as u



# This is the latitutde/longitude of the camera
cameraloc = (31.959417, -111.598583)

# I'm honestly guessing on the height right now.
cameraearth = EarthLocation(lat = cameraloc[0] * u.deg, lon = cameraloc[1] * u.deg, height = 2096 * u.m)

# Center of the circle found using super acccurate photoshop layering techniques.
center = (256, 252)

# The point for which we're doing this mathmagic.
point = (255, 252)

# Point adjusted based on the center being at... well... the center. And not the top left. In case you were confused.
pointadjust = (point[0] - center[0], point[1] - center[1])

# This divides by 0 if you're on the x axis so we have to just hardcode in those angles.
# This gets a bit weird if the exception isn't divide by zero.
try:
    az = math.atan((pointadjust[0])/(pointadjust[1]))
except:
    if pointadjust[0] < 0:
        az = math.pi / 2
    else:
        az = 3 * math.pi / 2
az = math.degrees(az)

# Add 180 degrees if the point is in the lower half of the image
# This is due to the way python takes the angles. tl;dr Top half measured from north, bottom half measured from south
if pointadjust[1] > 0:
    az += 180
# Add 360 in the top right quadrant.
elif pointadjust[0] > 0 and pointadjust[1] < 0:
    az += 380

# Pythagorean thereom boys.
r = math.sqrt(pointadjust[0]**2 + pointadjust[1]**2)

# For now if r is on the edge of the circle or beyond we'll have it just be 90 degrees.
if r >= 239:
    alt = 90
else:
    #r = 238 # Debug line
    r = r * 0.0483264 # Magic pixel to mm conversion rate


    # This is probably very not accurate right now. 
    # I used a table of r -> theta values for this lens and then found a regression for it
    a = 7.138255950800197
    b = 0.021488582426739067
    c = 0.0002825039064767284
    alt =  a*r + b*math.pow(r,2) + c*math.pow(r,4)

# The alt computed is down from the vertical, rather than up from the horizontal. Easy fix.
alt = 90 - alt

print('Azimuthal angle = ' + str(az))
print('Altitude angle = ' + str(alt))

# obstime?

time = Time('2017-7-12 23:00:00')

altazcoord = SkyCoord(alt = alt * u.deg, az = az * u.deg, frame = 'altaz', obstime = time, location = cameraearth)
radeccoord = altazcoord.icrs

print(radeccoord)