import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import math
import ephem


def daynum(day, format='%Y%m%d'):
    # Strips out the information from the pass in string
    d1 = datetime.date(int(day[:4]), int(day[4:6]), int(day[6:8]))

    # The date for the start of the year.
    d2 = datetime.date(year=d1.year, month=1, day=1)

    # Gotta add one because of how subtraction works.
    days = (d1-d2).days + 1
    return days


def format_date(day, name):
    formatdate = day[:4] + '/' + day[4:6] + '/' + day[6:8]
    time = name[4:6] + ':' + name[6:8] + ':' + name[8:10]
    return formatdate + ' ' + time


def find_threshold():
    # Sets up a pyephem object for the camera.
    camera = ephem.Observer()
    camera.lat = '31.959417'
    camera.lon = '-111.598583'
    camera.elevation = 2120
    camera.horizon = '-17'

    # Reads in the csv file using pandas.
    data = pd.read_csv('daily-2007-2017.csv')

    # This array is for calculating the total average
    total = []
    years = {}
    for year in ['2016', '2017']:
        # Gets the downloaded months
        directory = 'Data/'

        # If the data directory doesn't exist we should exit here.
        # I was remarkably prescient writing this even though I had all the data
        # downloaded already.
        if not os.path.exists(directory):
            print('No data found.')

        months = sorted(os.listdir(directory))

        # Macs are dumb
        if '.DS_Store' in months:
            months.remove('.DS_Store')

        # We do this as a dict because 2017 is straight up missing some days of
        # images because I guess the camera was down?
        # Otherwise I'd just make a len 365 list.
        daydict = {}
        for month in months:
            # If the month is not in this year, we skip it analyzing.
            if int(month) < int(year + '01') or int(month) > int(year + '12'):
                continue

            # Gets the days that were analyzed for that month
            directory = 'Data/' + month + '/'
            days = sorted(os.listdir(directory))

            # Macs are still dumb.
            if '.DS_Store' in days:
                days.remove('.DS_Store')

            # Reads the data for each day.
            for day in days:
                # Skip the leap day for right now.
                if day == '20160229.txt':
                    continue

                # Get the number for that day to add it to the dict.
                i = daynum(day)

                # Because we skip the leap day we need to bump the day num of all
                # days after that date down by one.
                # 60 because 31 + 28 = 59
                if year == '2016' and i >= 60:
                    i = i-1

                # Start with an empty list for that day.
                daydict[i] = []

                # This is the code that reads in the values and appends them.
                dataloc = directory + day
                datafile = open(dataloc, 'r')

                # Images is a list of images that were analyzed that night.
                images = []
                for line in datafile:
                    line = line.rstrip()
                    line = line.split(',')

                    # Appends the image name to images and the cloudiness relative
                    # to mean to the daydict.
                    images.append(line[0])
                    daydict[i].append(float(line[1]))

                # This block only runs if we actually analyzed images.
                if images:
                    # The first image analyzed that night.
                    name = images[0]
                    firstdate = format_date(day, name)
                    camera.date = firstdate
                    date1 = ephem.Date(firstdate)

                    # The final image analyzed that night.
                    name = images[-1]
                    date2 = ephem.Date(format_date(day, name))

                    # Calculates the previous setting and next rising of the sun
                    # using the first image analyzed that night.
                    sun = ephem.Sun()
                    setting = camera.previous_setting(sun, use_center=True)
                    rising = camera.next_rising(sun, use_center=True)

                    night = rising - setting
                    covered = date2 - date1


        # An ndarray of open fractions where index + 1 = day number
        opens = data.get('Y' + year).values
        thresh = []

        x = []
        x1 = []
        true = []
        # Runs over the dictionary, key is the day number. Val is the list of
        # cloudinesses
        for key, val in daydict.items():
            # The fraction is the fraction of the night the dome was closed.
            # When we multiply to find the index we want the inverse frac though.
            frac = 1 - opens[key - 1]

            # Finds the index at which the fraction of images above that index is
            # equal to the amount of the night that the dome was closed.
            working = sorted(val)

            # If we don't have any images that night then just bail.
            if len(working) == 0:
                continue

            # Multiply the frac by the length, to find the index above which
            # the correct fraction of the images is 'dome closed.' Rounds and
            # Subtracts one to convert it to the integer index.
            index = int(round(frac * len(working))) - 1

            # If the index is the final index then the 'cloudiness relative to the
            # mean threshold' is slightly below that value so average down.
            # Otherwise take the average of that index and the one above since the
            # threshold actually falls inbetween.
            if index == len(working) - 1 and not frac == 1:
                num = np.mean([float(working[index]), float(working[index - 1])])
            # If the dome is closed the entire night, index will be given as -1
            # And we find the threshold as the average of the start and end
            # cloudiness. Instead we want the threshold to be the first
            # cloudiness as that way the dome is "closed" all night.
            elif frac == 0:
                num = float(working[0]) - 0.1
            elif frac == 1:
                num = float(working[-1])
            else:
                num = np.mean([float(working[index]), float(working[index + 1])])

            thresh.append(num)
            total.append(num)
            x.append(key)

            working = np.asarray(working)
            above = working[working > num]

            if len(working) > 0:
                frac = opens[key - 1]
                true.append(len(above)/len(working))
                x1.append(frac)


        print(year + ': ' + str(np.median(thresh)))
        print()

        fig,ax = plt.subplots()
        fig.set_size_inches(6, 4)

        above = np.ma.masked_where(thresh < np.median(thresh), thresh)
        below = np.ma.masked_where(thresh > np.median(thresh), thresh)
        ax.scatter(x, below, s=1)
        ax.scatter(x, above, s=1, c='r')
        ax.set_xlabel('Day')
        ax.set_ylabel('Cloudiness Relative to Mean')
        plt.savefig('Images/Threshold-' + year + '.png', dpi=256)
        plt.close()

        fig,ax = plt.subplots()
        fig.set_size_inches(6, 4)
        ax.scatter(x1, true, s=1)
        ax.set_xlabel('True Fraction')
        ax.set_ylabel('Found Fraction')
        plt.savefig('Images/Differences-' + year + '.png', dpi=256)
        plt.close()

    #years[year] = daydict

    return np.median(total)


def test_threshold():
    for year in ['2016', '2017']:

        # Reads in the csv file using pandas.
        data = pd.read_csv('daily-2007-2017.csv')

        opens = data.get('Y' + year).values

        # Gets the downloaded months
        directory = 'Data/'

        # If the data directory doesn't exist we should exit here.
        # I was remarkably prescient writing this even though I had all the data
        # downloaded already.
        if not os.path.exists(directory):
            print('No data found.')

        months = sorted(os.listdir(directory))

        # Macs are dumb
        if '.DS_Store' in months:
            months.remove('.DS_Store')

        test = find_threshold()

        print(test)

        # We do this as a dict because 2017 is straight up missing some days of
        # images because I guess the camera was down?
        # Otherwise I'd just make a len 365 list.
        daydict = {}
        for month in months:
            # If the month is not in this year, we skip it analyzing.
            if int(month) < int(year + '01') or int(month) > int(year + '12'):
                continue

            # Gets the days that were analyzed for that month
            directory = 'Data/' + month + '/'
            days = sorted(os.listdir(directory))

            # Macs are still dumb.
            if '.DS_Store' in days:
                days.remove('.DS_Store')

            # Reads the data for each day.
            for day in days:
                # Skip the leap day for right now.
                if day == '20160229.txt':
                    continue

                # Get the number for that day to add it to the dict.
                i = daynum(day)

                # Because we skip the leap day we need to bump the day num of all
                # days after that date down by one.
                # 60 because 31 + 28 = 59
                if year == '2016' and i >= 60:
                    i = i-1

                # Start with an empty list for that day.
                daydict[i] = []

                # This is the code that reads in the values and appends them.
                dataloc = directory + day
                datafile = open(dataloc, 'r')

                # Images is a list of images that were analyzed that night.
                images = []
                for line in datafile:
                    line = line.rstrip()
                    line = line.split(',')

                    # Appends the image name to images and the cloudiness relative
                    # to mean to the daydict.
                    images.append(line[0])
                    daydict[i].append(float(line[1]))

            x = []
            true = []
            # Runs over the dictionary, key is the day number. Val is the list of
            # cloudinesses
            for key, val in daydict.items():
                # The fraction is the fraction of the night the dome was closed.
                # When we multiply to find the index we want the inverse frac though.
                frac = opens[key - 1]

                working = np.asarray(val)


                above = working[working > test]

                if len(working) > 0:
                    true.append(len(above)/len(working))
                    x.append(frac)

        fig,ax = plt.subplots()
        fig.set_size_inches(6, 4)
        ax.scatter(x, true, s=1)
        ax.set_xlabel('True Fraction')
        ax.set_ylabel('Found Fraction')
        plt.savefig('Images/Differences-' + year + '.png', dpi=256)
        plt.close()


find_threshold()