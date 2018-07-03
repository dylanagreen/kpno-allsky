import requests
from scipy import ndimage
from html.parser import HTMLParser
import matplotlib.pyplot as plt

import ImageIO
import Moon
import Histogram
import Coordinates

import os
import time


def get_start():
    # Gets the downloaded months
    directory = 'Data/'
    months = sorted(os.listdir(directory))

    # If no dates have been analyzed yet return 0
    if not months or months[-1] == '.DS_Store':
        return 0

    # Gets the singular days that have been analyzed
    directory = 'Data/' + months[-1] + '/'
    days = sorted(os.listdir(directory))

    # If no days for this month have been analyzed yet return the first day.
    if not days:
        start = months[-1] + '01'
        return start

    # Return the final day analyzed. We will reanalyze this completely,
    # in case the analysis crashed mid day.
    # Second slice slices off the .txt
    start = days[-1][:-4]
    return start


def analyze():
    t1 = time.perf_counter()
    link = 'http://kpasca-archives.tuc.noao.edu/'

    rlink = requests.get(link)

    html = rlink.text
    # This parser was designed to work with image names but it works just the same
    # For dates so why make a new one?
    parser = ImageIO.DateHTMLParser()
    parser.feed(html)
    parser.close()
    datelinks = parser.data
    parser.clear_data()

    # 20080306 is the first date the camera is located correctly
    # 20080430 is the first date the images are printed correctly

    #print(datelinks)

    startdate = int(get_start())

    for date in datelinks:

        # Strips the /index from the date so we have just the date
        d = date[:8]

        # The month of the date for organization purposes.
        month = d[:6]



        # This makes it so we start with the date we left off with.
        if int(d) < startdate:
            continue

        if not(20170101 <= int(d) <= 20171231):
            continue

        print(d)


        rdate = requests.get(link + date)

        # Extracting the image names.
        htmldate = rdate.text
        parser.feed(htmldate)
        parser.close()
        imagenames = parser.data
        parser.clear_data()

        #print(imagenames)

        monthloc = 'Data/' + month + '/'

        if not os.path.exists(monthloc):
            os.makedirs(monthloc)

        datafile = monthloc + d + '.txt'
        #f = open(datafile, 'w')

        with open(datafile, 'w') as f:

            for name in imagenames:
                # This ignores all the blue filter images, since we don't want
                # to process those.
                if name[:1] == 'b':
                    continue
                # We want to ignore the all image animations
                if name == 'allblue.gif' or name == 'allred.gif':
                    continue


                # Finds the moon and the sun in the image. We don't need to
                # download it
                # if we won't process it.
                # We only process images where the moon is visble (moon alt > 0)
                # And the sun is low enough to not wash out the image
                # (sun alt < -17)
                moonx, moony, moonalt = Moon.find_moon(d, name)
                sunalt, sunaz = Moon.find_sun(d, name)

                # Checks that the analysis conditions are met.
                if moonalt > 0 and sunalt < -17:
                    # Here is where the magic happens.
                    # First we download the image.
                    ImageIO.download_image(d, name)

                    # Then we make a histogram and a "cloudiness fraction"
                    path = 'Images/Original/KPNO/' + d + '/' + name
                    img = ndimage.imread(path, mode='L')

                    # Generates the moon mask.
                    mask = Moon.moon_mask(d, name)
                    hist, bins = Histogram.generate_histogram(img, mask)

                    frac = Histogram.cloudiness(hist)

                    # Then we save the cloudiness fraction to the file for that
                    # date.
                    dataline = name + ',' + str(frac) + '\n'
                    print(dataline)
                    f.write(dataline)
    t2 = time.perf_counter()
    
    print(t2-t1)


def plot():
    # Gets the downloaded months
    directory = 'Data/'
    months = sorted(os.listdir(directory))

    # Macs are dumb
    months.remove('.DS_Store')

    # Just plotting the last month analyzed right now
    month = months[-1]
    
    for month in months:
        # Gets the days that were analyzed for that month
        directory = 'Data/' + month + '/'
        days = sorted(os.listdir(directory))

        data = []
        illum = []
        x = []
        # Reads the data for each day.
        for day in days:
            d1 = directory + day
            f1 = open(d1, 'r')

            # Strips off the .txt so we can make a Time object.
            day = day[:-4]
            for line in f1:
                line = line.rstrip()
                line = line.split(',')

                # Gets the plot date for the timestring object.
                d = Coordinates.timestring_to_obj(day, line[0]).plot_date

                data.append(float(line[1]))
                x.append(d)
            
                vis = Moon.moon_visible(day, line[0])
                illum.append(vis)

            f1.close()

        # Sets up the plot
        fig, ax = plt.subplots()
        fig.set_size_inches(20,5)
        ax.plot_date(x, data, xdate=True, markersize=1)
    
        ax.plot_date(x, illum, xdate=True, markersize=1, color='red')

        # These are the x divisions which indicate the start of each day.
        xt = []
        start = int(days[0][:-4])
        end = int(days[-1][:-4]) + 1
        for i in range(start, end):
            # Finds the plot_date for the start of the day.
            x1 = Coordinates.timestring_to_obj(str(i), 'r_ut000000s00000').plot_date
            xt.append(x1)

        # Need the right end to be the start of next month.
        # Adding 100 increments the month number.
        end = str(start + 100)
        d2 = Coordinates.timestring_to_obj(end, 'r_ut000000s00000').plot_date

        # Sets the limits and division ticks.
        ax.set_ylim(0, 1.0)
        ax.set_xlim(xt[0], d2)
        ax.set_xticks(xt)
        ax.xaxis.grid(True)
        ax.xaxis.set_ticklabels([])

        # Sets the axis labels and saves.
        ax.set_ylabel('Cloudiness Fraction')
        
        # Puts the date in a nice form for the plot axis.
        date = month[-2:] + '/01/' + month[:-2]
        ax.set_xlabel('Time After ' + date)
        plt.savefig('Images/scatter-' + month + '.png', dpi=256, bbox_inches='tight')
        plt.close()
    


plot()