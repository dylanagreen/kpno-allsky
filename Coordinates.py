import math

# Center of the circle found using super acccurate photoshop layering techniques.
center = (256, 252)

point = (296, 212)

# Point adjusted based on the center being at... well... the center.
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
# This is due to the way python takes the angles.
if pointadjust[1] > 0:
    az += 180
# Add 360 in the top right quadrant.
elif pointadjust[0] > 0 and pointadjust[1] < 0:
    az += 380

r = math.sqrt(pointadjust[0]*pointadjust[0] + pointadjust[1]*pointadjust[1])
r = 239
r = r * 0.0483264 # Magic pixel to mm conversion rate


# This is probably very not accurate right now. 
# I used a table of r -> theta values for this lens and then found a regression for it
alt =  - 309.7930342198971 + 310.1190311630313 * math.exp(0.021947630111875834 * r)

print('Azimuthal angle = ' + str(az))
print('Altitude angle = ' + str(alt))