import os
from PIL import Image as pil_image
from astropy.time import Time
import numpy as np


class AllSkyImage():
    def __init__(self, name, date, camera, data):
        self.name = name
        self.date = date
        self.camera = camera
        self.data = data
        
        format1 = date[:4] + '-' + date[4:6] + '-' + date[6:8]
        format2 = name[4:6] + ':' + name[6:8] + ':' + name[8:10]
        self.formatdate = format1 + ' ' + format2
        
        self.time = Time(self.formatdate)
        
        


def load_image(name, date, camera):
    # If the name was passed without .png at the end append it so we know what
    # format this bad boy is in.
    if not name[-4:] == '.png':
        name = name + '.png'
    
    # Loads the image using Pillow and converts it to greyscale
    loc = os.path.join('Images', *['Original', camera, date, name])
    img = np.asarray(pil_image.open(loc).convert('L'))
    return AllSkyImage(name, date, camera, img)
    
if __name__ == "__main__":
    test = load_image('r_ut005728s27480', '20160101', 'KPNO')
    print(test.time)