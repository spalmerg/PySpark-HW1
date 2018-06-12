from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == "__main__":
  # set up drivers
  sc = SparkContext(appName='Question 1')
  sqlContext = SQLContext(sc)

  # read in data and register as querable database
  file = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true') \
   .load('hdfs://wolf.iems.northwestern.edu/user/sgreenwood/crime/Crimes_-_2001_to_present.csv')
  file.registerTempTable("crimes")

  # query all dates of crimes
  dates = sqlContext.sql("SELECT SUBSTR(Date, 1, 2) AS Month, \
                        SUBSTR(Date,7, 4) AS Year FROM crimes")

  # count number of crimes each month/year
  agg = dates.groupBy(["Month", "Year"]).count()
  agg = agg.toPandas()
  agg.columns = ['Month', 'Year', 'Tally']

  # get average crimes per month
  agg = agg.groupby("Month").mean()

  # generate plot
  fig = agg.plot.bar(title = "Average Crime Count Per Month", rot = 0)
  fig.set_ylabel("Average Number of Crimes")
  final = fig.get_figure()
  final.savefig("exercise1.pdf")

  # close spark session
  sc.stop()