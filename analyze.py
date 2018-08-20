import os
import time
import ast
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ephem

import io_util
import moon
import histogram
import coordinates


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

    rlink = io_util.download_url(link)

    if rlink is None:
        print('Getting dates failed.')
        return

    html = rlink.text
    # This parser was designed to work with image names but it works the same
    # for dates so why make a new one?
    parser = io_util.DateHTMLParser()
    parser.feed(html)
    parser.close()
    datelinks = parser.data
    parser.clear_data()

    # 20080306 is the first date the camera is located correctly
    # 20080430 is the first date the images are printed correctly

    startdate = 0

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

        rdate = io_util.download_url(link + date)

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
                # We only process images where the moon is visble (alt > 0)
                # And the sun is low enough to not wash out the image
                # (sun alt < -17)
                moonalt = moon.find_moon(d, name)[2]
                sunalt = moon.find_sun(d, name)[0]

                # Checks that the analysis conditions are met.
                if moonalt > 0 and sunalt < -17:
                    # This is the path the image is saved to.
                    # We need to check if it exists first, just in case.
                    path = 'Images/Original/KPNO/' + d + '/' + name

                    # Here is where the magic happens.
                    # First we download the image, if it hasn't been downloaded
                    if not os.path.isfile(path):
                        io_util.download_image(d, name)

                    # Then we make a histogram and a "cloudiness fraction"
                    img = ndimage.imread(path, mode='L')

                    # Generates the moon mask.
                    mask = moon.moon_mask(d, name)
                    bins = histogram.generate_histogram(img, mask)[1]

                    frac = histogram.cloudiness(hist)

                    # Correction for moon phase.
                    phase = moon.moon_visible(d, name)
                    val = b*phase*phase + c*phase

                    with open('values.txt', 'a') as f2:
                        towrite = str(phase) + ',' + str(val) + ',' + str(frac)
                        f2.write(towrite + '\n')

                    frac = frac/val

                    # Then we save the cloudiness fraction to the file for that
                    # date.
                    dataline = name + ',' + str(frac) + '\n'
                    print(dataline)
                    f.write(dataline)
    t2 = time.perf_counter()

    print(t2 - t1)

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
                d = coordinates.timestring_to_obj(day, line[0]).plot_date

                data.append(float(line[1]))
                x.append(d)

                vis = moon.moon_visible(day, line[0])
                illum.append(vis)

            f1.close()

        # Sets up the plot
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 5)
        ax.plot_date(x, data, xdate=True, markersize=1)

        ax.plot_date(x, illum, xdate=True, markersize=1, color='red')

        # These are the x divisions which indicate the start of each day.
        xt = []
        start = int(days[0][:-4])
        end = int(days[-1][:-4]) + 1
        for i in range(start, end):
            # Finds the plot_date for the start of the day.
            x1 = coordinates.timestring_to_obj(str(i),
                                               'r_ut000000s00000').plot_date
            xt.append(x1)

        # Need the right end to be the start of next month.
        # Adding 100 increments the month number.
        # First if statement checks if this is december in which case we need
        # to roll over the year and reset the month.
        if str(start)[4:6] == '12':
            end = str(start + 8900)
        else:
            end = str(start + 100)
        d2 = coordinates.timestring_to_obj(end, 'r_ut000000s00000').plot_date

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

        plt.savefig(plotloc + 'scatter-' + month + '.png',
                    dpi=256, bbox_inches='tight')
        plt.close()


# Sets up a pyephem object for the camera.
camera = ephem.Observer()
camera.lat = '31.959417'
camera.lon = '-111.598583'
camera.elevation = 2120
camera.horizon = '-17'


def plot():
    # Gets the downloaded months
    directory = 'Data/'

    # If the data directory doesn't exist we should exit here.
    if not os.path.exists(directory):
        print('No data found.')
        return

    # Temporary arrays for moon phase plots
    phasenum = 50
    phasediv = 1 / phasenum  # Phase ranges from 0-1 so divisions is 1/num divs
    tphase = [[] for i in range(0, phasenum)]

    # Temporary arrays for sunset plots
    sunsetnum = 50
    sunsetdiv = 0.5 / sunsetnum  # Up to 12 hours after sunset = 0.5 day / divs
    tsunset = [[] for i in range(0, sunsetnum)]

    # Temporary arrays for sunrise plots
    sunrisenum = 50
    sunrisediv = 0.5 / sunrisenum  # 12 hours before sunrise = 0.5 day / divs
    tsunrise = [[] for i in range(0, sunrisenum)]

    # Temp other stuff
    normalnum = 50
    normaldiv = 1. / normalnum
    tnormal = [[] for i in range(0, normalnum)]

    # Temporary arrays for week plots
    tweek = [[] for i in range(0, 53)]
    tmoon = [[] for i in range(0, 53)]

    months = sorted(os.listdir(directory))

    # Macs are dumb
    if '.DS_Store' in months:
        months.remove('.DS_Store')

    for month in months:
        # Gets the days that were analyzed for that month
        directory = 'Data/' + month + '/'
        days = sorted(os.listdir(directory))

        # Strips out the year from the month
        year = month[:4]

        # Day 1 of the year, for week calculation.
        yearstart = year + '0101'
        day1 = coordinates.timestring_to_obj(yearstart, 'r_ut000000s00000')

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
                phase = moon.moon_visible(day, name)

                # Ignores cloudiness with moon phase less than 0.2
                if phase < 0.2:
                    continue

                b = int(phase // phasediv)
                tphase[b].append(val)

                # Sunset time calculation.
                # 12 hours after sunset for 50 bins = 0.01 of a day per bin.
                formatdate = day[:4] + '/' + day[4:6] + '/' + day[6:]
                time = name[4:6] + ':' + name[6:8] + ':' + name[8:10]
                formatdate = formatdate + ' ' + time

                # Sets the date of calculation.
                camera.date = formatdate
                date = ephem.Date(formatdate)

                # Calculates the previous setting and next rising of the sun.
                sun = ephem.Sun()
                setting = camera.previous_setting(sun, use_center=True)
                rising = camera.next_rising(sun, use_center=True)

                # Finds the difference and bins it into the correct bin.
                diff = date - setting
                length = rising - setting
                b = int(diff // sunsetdiv)
                b2 = int((diff / length) // normaldiv)
                tsunset[b].append(val)
                tnormal[b2].append(val)

                # Finds the difference and bins it into the correct bin.
                diff = rising - date
                b = int(diff // sunrisediv)
                tsunrise[b].append(val)

                # Week of the year calculation.
                # Gets the date object for the image
                date = coordinates.timestring_to_obj(day, name)

                # Finds the difference since the beginning of the year to find
                # The week number.
                diff = date - day1
                week = int(diff.value // 7)
                tweek[week].append(val)
                tmoon[week].append(phase)

    percents = ['25', '50', '75']
    colors = [(1, 0, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1)]

    # Nan array for if there's no data for that bin.
    nanarray = np.asarray([[float('nan'), float('nan'), float('nan')]])

    # This method sets up the plots. using the given x array and the data set
    # it finds the percentile values for each bin division.
    def setup_plot(x, dataset):
        # We only need this so the axis shape works out. We delete it later.
        data = np.asarray([[0, 0, 0]])

        for i, val in enumerate(dataset):
            temp = np.asarray(val)

            # Percentile returns an array of each of the three values.
            # We're creating a data array where each column is all the
            # percentile values for each bin value.
            if temp.size == 0:
                data = np.append(data, nanarray, axis=0)
            else:
                d = np.reshape(np.percentile(temp, [25, 50, 75]), (1, 3))
                data = np.append(data, d, axis=0)

        # Deletes the first 0,0,0 array.
        data = np.delete(data, 0, 0)

        plt.ylim(0, 4.0)

        # This sets up the individual plots. You can just pass the data
        # multiararay but I need each line to be individually labeled.
        for i, val in enumerate(percents):
            plt.plot(x, data[0:data.shape[0], i], label=val + '%',
                     color=colors[i])

        # This fills between the lowest and top percentiles.
        plt.fill_between(x, data[0:data.shape[0], 0],
                         data[0:data.shape[0], -1], color=(1, 0.5, 0, 0.5))
        plt.legend()

    # Plotting and averaging code
    # Moon phase
    plt.ylabel('Cloudiness Relative to Mean')
    plt.xlabel('Moon Phase')

    x = np.asarray((range(0, phasenum)))
    x = x * phasediv + (phasediv / 2.)
    setup_plot(x, tphase)
    plt.savefig('Images/Plots/phase.png', dpi=256, bbox_inches='tight')
    plt.close()

    # Sunset
    plt.ylabel('Cloudiness Relative to Mean')
    plt.xlabel('Hours since sunset')

    x = np.asarray((range(0, sunsetnum)))
    x = x * sunsetdiv * 24 + (sunsetdiv * 12)
    setup_plot(x, tsunset)
    plt.savefig('Images/Plots/sunset.png', dpi=256, bbox_inches='tight')
    plt.close()

    # Normalized
    plt.ylabel('Cloudiness Relative to Mean')
    plt.xlabel('Normalized time after sunset')

    x = np.asarray((range(0, normalnum)))
    x = x * normaldiv + (normaldiv / 2.)
    setup_plot(x, tnormal)
    plt.savefig('Images/Plots/normalized-0.png', dpi=256, bbox_inches='tight')
    plt.close()

    # Sunrise
    x = np.asarray((range(0, sunrisenum)))
    x = x * sunrisediv * 24 + (sunrisediv * 12)

    # Sets up the plot before we plot the things
    plt.ylabel('Cloudiness Relative to Mean')
    plt.xlabel('Hours before sunrise')

    setup_plot(x, tsunrise)
    plt.savefig('Images/Plots/sunrise.png', dpi=256, bbox_inches='tight')
    plt.close()

    # Week
    x = np.asarray((range(1, 54)))
    plt.ylabel('Cloudiness Relative to Mean')
    plt.xlabel('Week Number')

    # Moon phase averages.
    moon_avgs = []
    num_imgs = []
    for i, val in enumerate(tweek):
        moon_avgs.append(np.mean(tmoon[i]))
        print(str(i + 1) + ': ' + str(len(val)))
        num_imgs.append(len(val))

    setup_plot(x, tweek)
    # plt.plot(x, moons, label='Moon Phase', color=(0, 1, 0, 1))

    num_imgs = np.asarray(num_imgs)

    num_imgs = num_imgs * 1 / (np.amax(num_imgs))

    plt.plot(x, num_imgs, label='Normalized number of images',
             color=(0, 1, 0, 1))

    # We have to re add the legend to get the moon phase label.
    plt.legend()
    plt.savefig('Images/Plots/week.png', dpi=256, bbox_inches='tight')
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
        b, c = np.linalg.lstsq(A, data[i])[0]

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


def histo():
    directory = 'Data/'

    months = sorted(os.listdir(directory))

    # Macs are dumb
    if '.DS_Store' in months:
        months.remove('.DS_Store')

    tweek = [[] for i in range(0, 53)]
    tweek2 = [[] for i in range(0, 53)]

    for month in months:

        # Gets the days that were analyzed for that month
        directory = 'Data/' + month + '/'
        days = sorted(os.listdir(directory))

        # Strips out the year from the month
        year = month[:4]

        # Day 1 of the year, for week calculation.
        yearstart = year + '0101'
        day1 = coordinates.timestring_to_obj(yearstart, 'r_ut000000s00000')

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
                phase = moon.moon_visible(day, name)

                # Ignores cloudiness with moon phase less than 0.2
                if phase < 0.2:
                    continue

                date = coordinates.timestring_to_obj(day, name)

                # Finds the difference since the beginning of the year to find
                # The week number.
                diff = date - day1
                week = int(diff.value // 7)
                tweek[week].append(val)

                if year == '2016':
                    tweek2[week].append(val)

    w = 0.61

    # Starts by finding the divs because we want the width to be the same.
    num = np.amax(tweek[2]) / w
    divs = np.asarray(range(0, int(num) + 1))
    divs = divs * w

    saveloc = 'Images/Plots/Weeks'
    if not os.path.exists(saveloc):
        os.makedirs(saveloc)

    for i, val in enumerate(tweek):
        # Finds the histograms.
        hist1, bins1 = np.histogram(val, bins=divs)
        hist2, bins2 = np.histogram(tweek2[i], bins=divs)

        n2 = len(tweek2[i])
        n1 = len(val) - n2

        # Sets the size wider than th eprevious to fit all the bins.
        # I shave off a lot of 0 value bins later as well (in the plotting
        # slice)
        fig = plt.figure()
        fig.set_size_inches(11.4, 8.4)

        # Plotting code.
        plt.title('Week ' + str(i+1))
        plt.ylim(0, 900)
        plt.ylabel('Number of Occurrences')
        plt.xlabel('Cloudiness Relative to Mean')
        plt.bar(bins1[:-16], hist1[:-15], width=w, align='edge',
                tick_label=bins1[:-16], label='2017 (' + str(n1) + ')')

        plt.bar(bins2[:-16], hist2[:-15], width=w, align='edge',
                tick_label=bins2[:-16], color='red',
                label='2016 (' + str(n2) + ')')

        plt.legend()
        plt.savefig('Images/Plots/Weeks/hist-' + str(i + 1) + '.png',
                    dpi=256, bbox_inches='tight')
        plt.close()

        print('Saved: Week ' + str(i+1))


def to_csv():
    directory = 'Data/'

    months = sorted(os.listdir(directory))

    # Macs are dumb
    if '.DS_Store' in months:
        months.remove('.DS_Store')

    # Temp other stuff
    normalnum = 50
    normaldiv = 1. / normalnum

    data = []

    for month in months:

        # Gets the days that were analyzed for that month
        directory = 'Data/' + month + '/'
        days = sorted(os.listdir(directory))

        # Strips out the year from the month
        year = month[:4]

        # Day 1 of the year, for week calculation.
        yearstart = year + '0101'
        day1 = coordinates.timestring_to_obj(yearstart, 'r_ut000000s00000')

        # Reads the data for each day.
        for day in days:
            loc = directory + day
            f1 = open(loc, 'r')

            # Strips off the .txt so we can make a Time object.
            day = day[:-4]
            for line in f1:
                linedata = []

                # Splits out the value and file.
                line = line.rstrip()
                line = line.split(',')
                val = float(line[1])
                name = line[0]

                # Moon phase calculation.
                phase = moon.moon_visible(day, name)

                # Ignores cloudiness with moon phase less than 0.2
                if phase < 0.2:
                    continue

                date = coordinates.timestring_to_obj(day, name)

                # Finds the difference since the beginning of the year to find
                # The week number.
                diff = date - day1
                week = int(diff.value // 7)

                # Sunset time calculation.
                # 12 hours after sunset for 50 bins = 0.01 of a day per bin.
                formatdate = day[:4] + '/' + day[4:6] + '/' + day[6:]
                time = name[4:6] + ':' + name[6:8] + ':' + name[8:10]
                formatdate = formatdate + ' ' + time

                linedata.append(formatdate)
                linedata.append(year)
                linedata.append(week)

                # Sets the date of calculation.
                camera.date = formatdate
                date = ephem.Date(formatdate)

                # Calculates the previous setting and next rising of the sun.
                sun = ephem.Sun()
                setting = camera.previous_setting(sun, use_center=True)
                rising = camera.next_rising(sun, use_center=True)

                # Finds the difference and bins it into the correct bin.
                diff = date - setting
                length = rising - setting
                normalize = int((diff / length) // normaldiv)

                linedata.append(normalize)
                linedata.append(val)

                data.append(np.asarray(linedata))

    data = np.asarray(data)

    d2 = pd.DataFrame(data, columns=['Date & Time', 'Year', 'Week Number',
                                     'Normlaized Time after Sunset',
                                     'Cloudiness Relative to the Mean'])

    d2.to_csv('data.csv')


if __name__ == "__main__":
    # This link has a redirect loop for testing.
    # link = 'https://demo.cyotek.com/features/redirectlooptest.php'
    plot()