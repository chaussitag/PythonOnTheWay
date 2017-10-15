#!/usr/bin/env python
# coding=utf8

import collections
import json
import pandas
import matplotlib.pyplot as plt

class StatisticsWithoutPandas(object):
    @staticmethod
    def getItemCounts(sequence):
        counts = collections.defaultdict(int)
        for item in sequence:
            counts[item] += 1
        return counts

    @staticmethod
    def topCountsItems(countsDict, n):
        valueKeyPairs = [(count, timeZone) for timeZone, count in countsDict.items()]
        valueKeyPairs.sort()
        return valueKeyPairs[-n:]

    @staticmethod
    def topTimeZones(records, n = 10):
        """Find top n popular time zone in records.

        Using just dict and list operations to find out top n popular time zones.

        Args:
            records (list of json objects): Data records in form of list, each item is a json object.
            n (Optional[int]): Defaults to 10.

        Returns:
            Top n time zones occurred in records in form of list, each element is a tuple of (time-zone, times_occurred)
        """
        timeZones = [item["tz"] for item in records if "tz" in item]
        countsDict = StatisticsWithoutPandas.getItemCounts(timeZones)
        topTimeZones = StatisticsWithoutPandas.topCountsItems(countsDict, n)
        print(topTimeZones)
        return topTimeZones

    @staticmethod
    def topTimeZonesByCounter(records, n = 10):
        """Find top n popular time zone in records.

        Using collections.Counter to find out top n popular time zones.

        Args:
            records (list of json objects): Data records in form of list, each item is a json object.
            n (Optional[int]): Defaults to 10.

        Returns:
            Top n time zones occurred in records in form of list, each element is a tuple of (time-zone, times_occurred)
        """
        timeZones = [item["tz"] for item in records if "tz" in item]
        counter = collections.Counter(timeZones)
        topTimeZones = counter.most_common(n)
        print(topTimeZones)
        return topTimeZones

class StatisticsWithPandas(object):
    @staticmethod
    def topTimeZones(records, n = 10):
        """Find and plot top n popular time zone in records

        Using pandas and matplotlib.pyplot to find out top n popular time zones and plot the result.

        Args:
            records (list of json objects): Data records in form of list, each item is a json object.
            n (Optional[int]): Defaults to 10.

        Returns:
            Top n time zones occurred in records in form of list, each element is a tuple of (time-zone, times_occurred)
        """
        dataFrame = pandas.DataFrame(records)
        fixedTimeZones = dataFrame["tz"].fillna("N/A")
        fixedTimeZones[fixedTimeZones == ''] = "Null"
        topTimeZones = fixedTimeZones.value_counts()
        print(topTimeZones[:10])
        topTimeZones[:10].plot(kind = "barh", rot = 0)
        plt.show()
        return topTimeZones[:10]


if __name__ == "__main__":
    dataPath = "../pydata-book.git/ch02/usagov_bitly_data2012-03-16-1331923249.txt"
    with open(dataPath, "r") as dataFile:
        dataRecords = [json.loads(line) for line in dataFile]
    StatisticsWithoutPandas.topTimeZones(dataRecords)
    StatisticsWithoutPandas.topTimeZonesByCounter(dataRecords)

    StatisticsWithPandas.topTimeZones(dataRecords)