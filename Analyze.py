import requests
from scipy import ndimage
from html.parser import HTMLParser
import matplotlib.pyplot as plt
import numpy as np
from requests.exceptions import TooManyRedirects, HTTPError, ConnectionError, Timeout, RequestException

import ImageIO
import Moon
import Histogram
import Coordinates

import os
import time
import ephem
import ast


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


# Reads a link, with exception handling and error checking built in.
# Returns a requests.Response object if it succeeds, returns None if it fails.
def download_url(link):

    tries = 0
    read = False

    while not read:
        try:
            # Tries to connect for 5 seconds.
            data = requests.get(link, timeout=5)

            # Raises the HTTP error if it occurs.
            data.raise_for_status()

            read = True
        # Too many redirects is when the link redirects you too much.
        except TooManyRedirects:
            print('Too many redirects.')
            return None
        # HTTPError is an error in the http code.
        except HTTPError:
            print('HTTP error with status code ' + str(data.status_code))
            return None
        # This is a failure in the connection unrelated to a timeout.
        except ConnectionError:
            print('Failed to establish a connection to the link.')
            return None
        # Timeouts are either server side (too long to respond) or client side
        # (when requests doesn't get a response before the timeout timer is up)
        # I have set the timeout to 5 seconds
        except Timeout:
            tries += 1

            if tries >= 3:
                print('Timed out after three attempts.')
                return None

            # Tries again after 5 seconds.
            time.sleep(5)

        # Covers every other possible exceptions.
        except RequestException as err:
            print('Unable to read link')
            print(err)
            return None
        else:
            print(link + ' read with no errors.')
            return data


def analyze():
    t1 = time.perf_counter()
    link = 'http://kpasca-archives.tuc.noao.edu/'

    rlink = download_url(link)

    if rlink is None:
        print('Getting dates failed.')
        return

    html = rlink.text
    # This parser was designed to work with image names but it works the same
    # for dates so why make a new one?
    parser = ImageIO.DateHTMLParser()
    parser.feed(html)
    parser.close()
    datelinks = parser.data
    parser.clear_data()

    # 20080306 is the first date the camera is located correctly
    # 20080430 is the first date the images are printed correctly
    #print(datelinks)

    startdate = 0#int(get_start())

    # Reads in the model coefficients.
    with open('clouds.txt', 'r') as f:
        for line in f:
            line = line.rstrip()
            line = line.split(',')
            b = float(line[0])
            c = float(line[1])

    fails = []
    for date in datelinks:

        # Strips the /index from the date so we have just the date
        d = date[:8]

        # The month of the date for organization purposes.
        month = d[:6]

        # This makes it so we start with the date we left off with.
        if int(d) < startdate:
            continue

        if not(20160101 <= int(d) <= 20171231):
            continue

        print(d)

        rdate = download_url(link + date)

        # If we fail to get the data for the date, log it and move to the next.
        if rdate is None:
            fails.append(date)
            print('Failed to retrieve data for ' + date)
            continue

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
                    # This is the path the image is saved to.
                    # We need to check if it exists first, just in case.
                    path = 'Images/Original/KPNO/' + d + '/' + name

                    # Here is where the magic happens.
                    # First we download the image, if it hasn't been downloaded.
                    if not os.path.isfile(path):
                        ImageIO.download_image(d, name)

                    # Then we make a histogram and a "cloudiness fraction"
                    img = ndimage.imread(path, mode='L')

                    # Generates the moon mask.
                    mask = Moon.moon_mask(d, name)
                    hist, bins = Histogram.generate_histogram(img, mask)

                    frac = Histogram.cloudiness(hist)

                    # Correction for moon phase.
                    phase = Moon.moon_visible(d, name)
                    val = b*phase*phase + c*phase

                    with open('values.txt', 'a') as f2:
                        f2.write(str(phase) + ',' + str(val) + ',' + str(frac) + '\n')

                    frac = frac/val

                    # Then we save the cloudiness fraction to the file for that
                    # date.
                    dataline = name + ',' + str(frac) + '\n'
                    print(dataline)
                    f.write(dataline)
    t2 = time.perf_counter()

    print(t2-t1)

    print('The following dates failed to download: ' + str(fails))


def month_plot():
    # Gets the downloaded months
    directory = 'Data/'

    # If the data directory doesn't exist we should exit here.
    if not os.path.exists(directory):
        print('No data found.')
        return

    months = sorted(os.listdir(directory))

    # Macs are dumb
    if '.DS_Store' in months:
        months.remove('.DS_Store')

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
        # First if statement checks if this is december in which case we need
        # to roll over the year and reset the month.
        if str(start)[4:6] == '12':
            end = str(start + 8900)
        else:
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

        plotloc = 'Images/Plots/'
        if not os.path.exists(plotloc):
            os.makedirs(plotloc)

        plt.savefig(plotloc + 'scatter-' + month + '.png', dpi=256, bbox_inches='tight')
        plt.close()


# Sets up a pyephem object for the camera.
camera = ephem.Observer()
camera.lat = '31.959417'
camera.lon = '-111.598583'
camera.elevation = 2120
camera.horizon='-17'


def plot():
    # Gets the downloaded months
    directory = 'Data/'

    # If the data directory doesn't exist we should exit here.
    if not os.path.exists(directory):
        print('No data found.')
        return

    months = sorted(os.listdir(directory))

    # Macs are dumb
    if '.DS_Store' in months:
        months.remove('.DS_Store')

    # Gets the years and removes the duplicates by converting to a set .
    years = set([x[:4] for x in months])

    # Temporary arrays for moon phase plots
    phasenum = 50
    phasediv = 1 / phasenum  # Phase ranges from 0-1 so divisions is 1/num divs
    tphase = {}
    #tphase2 = {}

    # Temporary arrays for sunset plots
    sunsetnum = 50
    sunsetdiv = 0.5 / sunsetnum # Up to 12 hours after sunset = 0.5 day / divs
    tsunset = {}
    #tsunset2 = {}

    # Temporary arrays for week plots
    tweek = {}
    #tweek2 = {}

    tmoon = {}

    for year in years:
        tphase[year] = [[] for i in range(0, phasenum)]
        tsunset[year] = [[] for i in range(0, sunsetnum)]
        tweek[year] = [[] for i in range(0, 53)]

        tmoon[year] = [[] for i in range(0, 53)]

    for month in months:
        # Gets the days that were analyzed for that month
        directory = 'Data/' + month + '/'
        days = sorted(os.listdir(directory))

        # Strips out the year from the month
        year = month[:4]

        # Day 1 of the year, for week calculation.
        yearstart = year + '0101'
        day1 = Coordinates.timestring_to_obj(yearstart, 'r_ut000000s00000')

        # Reads the data for each day.
        for day in days:
            loc = directory + day
            f1 = open(loc, 'r')

            # Strips off the .txt so we can make a Time object.
            day = day[:-4]
            for line in f1:
                # Splits out the value and file.
                line = line.rstrip()
                line = line.split(',')
                val = float(line[1])
                name = line[0]

                # Moon phase calculation.
                phase = Moon.moon_visible(day, name)

                # Ignores cloudiness with moon phase less than 0.2
                if phase < 0.2:
                    continue

                b = int(phase // phasediv)
                tphase[year][b].append(val)

                # Sunset time calculation.
                # 12 hours after sunset for 50 bins = 0.01 of a day per bin.
                formatdate = day[:4] + '/' + day[4:6] + '/' + day[6:]
                time = name[4:6] + ':' + name[6:8] + ':' + name[8:10]
                formatdate = formatdate + ' ' + time

                # Sets the date of calculation.
                camera.date = formatdate
                date = ephem.Date(formatdate)

                # Calculates the previous setting of the sun.
                sun = ephem.Sun()
                setting = camera.previous_setting(sun, use_center=True)

                # Finds the difference and bins it into the correct bin.
                diff = date - setting
                b = int(diff // sunsetdiv)
                tsunset[year][b].append(val)

                # Week of the year calculation.
                # Gets the date object for the image
                date = Coordinates.timestring_to_obj(day, name)

                # Finds the difference since the beginning of the year to find
                # The week number.
                diff = date - day1
                week = int(diff.value // 7)
                tweek[year][week].append(val)

                vis = Moon.moon_visible(day, name)
                tmoon[year][week].append(vis)

    # Plotting and averaging code
    # Moon phase
    data = {}

    x = np.asarray((range(0,phasenum)))
    x = x * phasediv

    # Sets up the plot before we plot the things
    plt.ylim(0, 4.0)
    plt.ylabel('Cloudiness Relative to Mean')
    plt.xlabel('Moon Phase')

    percents = ['25', '50', '75']

    # Nan array for if there's no data for that bin.
    nanarray = np.asarray([[float('nan'), float('nan'), float('nan')]])

    for year in years:
        # We only need this so the axis shape works out. We delete it later.
        data = np.asarray([[0, 0, 0]])

        for i in range(0,len(tphase[year])):
            temp = np.asarray(tphase[year][i])

            if temp.size == 0:
                data = np.append(data, nanarray, axis=0)
            else:
                d = np.reshape(np.percentile(temp, [25,50,75]), (1,3))
                data = np.append(data, d, axis=0)

        # Deletes the first 0,0,0 array.
        data = np.delete(data, 0, 0)
        for i in range(0, len(percents)):
            plt.plot(x, data[0:data.shape[0], i], label=percents[i] + '% - ' + year)

    plt.legend()
    plt.savefig('Images/Plots/phase.png', dpi=256, bbox_inches='tight')
    plt.close()

    # Sunset
    x = np.asarray((range(0,sunsetnum)))
    x = x * sunsetdiv * 24

    # Sets up the plot before we plot the things
    plt.ylim(0, 4.0)
    plt.ylabel('Cloudiness Relative to Mean')
    plt.xlabel('Hours since sunset')

    for year in years:
        data = np.asarray([[0, 0, 0]])

        for i in range(0,len(tsunset[year])):
            temp = np.asarray(tsunset[year][i])

            if temp.size == 0:
                data = np.append(data, nanarray, axis=0)
            else:
                d = np.reshape(np.percentile(temp, [25,50,75]), (1,3))
                data = np.append(data, d, axis=0)

        data = np.delete(data, 0, 0)
        for i in range(0, len(percents)):
            plt.plot(x, data[0:data.shape[0], i], label=percents[i] + '% - ' + year)

    plt.legend()
    plt.savefig('Images/Plots/sunset.png', dpi=256, bbox_inches='tight')
    plt.close()

    # Week
    for year in years:

        x = np.asarray((range(1,54)))
        # Sets up the plot before we plot the things
        plt.ylim(0, 4.0)
        plt.ylabel('Cloudiness Relative to Mean')
        plt.xlabel('Week Number')

        data = np.asarray([[0, 0, 0]])
        moon = []

        for i in range(0,len(tweek[year])):
            temp = np.asarray(tweek[year][i])

            if temp.size == 0:
                data = np.append(data, nanarray, axis=0)
            else:
                d = np.reshape(np.percentile(temp, [25,50,75]), (1,3))
                data = np.append(data, d, axis=0)

            moon.append(np.mean(tmoon[year][i]))

        data = np.delete(data, 0, 0)

        for i in range(0, len(percents)):
            plt.plot(x, data[0:data.shape[0], i], label=percents[i] + '% - ' + year)

        plt.plot(x, moon, label='Moon Phase-' + year)

        plt.legend()
        plt.savefig('Images/Plots/week' + year + '.png', dpi=256, bbox_inches='tight')
        plt.close()


def model():
    loc = 'phase.txt'

    data = []

    # Reads in the phase and cloudiness information.
    # Literal eval makes it so that it reads the list as a list rather than
    # a string.
    with open(loc, 'r') as f:
        for line in f:
            line = line.rstrip()
            b = ast.literal_eval(line)

            data.append(b)
    x = data[0]
    x.pop(0)
    x = np.asarray(x)


    # Just need to convert to floats including the nan.
    for i in range(1, len(data)):
        data[i] = [float(j) for j in data[i]]

    plt.ylim(0, 1.0)
    plt.ylabel('Average Cloudiness Fraction')
    plt.xlabel('Moon Phase')


    coeffs1 = []
    coeffs2 = []
    for i in range(1, len(data)):
        data[i].pop(0)

        # This line is to avoid hard coding in case I add more years of data.
        year = 2017 - len(data) + 1 + i

        # We do the fitting manually using the least squares algorithm.
        # This forces the fit to go through 0,0 since we do not pass a constant
        # coefficient, only the x^2 and x terms.
        A = np.vstack([x*x, x]).T
        b,c = np.linalg.lstsq(A, data[i])[0]

        coeffs1.append(b)
        coeffs2.append(c)

        plt.plot(x, data[i], label='Mean-' + str(year))
        plt.plot(x, b*x*x + c*x, label='Fit-' + str(year))

    # An average of the year wise fits.
    m = np.mean(coeffs1)
    n = np.mean(coeffs2)
    plt.plot(x, m*x*x + n*x, label='Mean Fit')

    # Writes the coefficients to a file for later use.
    with open('clouds.txt', 'w') as f:
        f.write(str(m) + ',' + str(n))

    plt.legend()
    plt.savefig('Images/temp.png', dpi=256, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # This link has a redirect loop for testing.
    #link = 'https://demo.cyotek.com/features/redirectlooptest.php'
    plot()