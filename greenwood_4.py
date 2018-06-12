from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd
from pyspark.sql.functions import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == "__main__":
  # set up drivers
  sc = SparkContext(appName='Question 4')
  sqlContext = SQLContext(sc)

  # read in dataset
  file = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true') \
   .load('hdfs://wolf.iems.northwestern.edu/user/sgreenwood/crime/Crimes_-_2001_to_present.csv')
  file.registerTempTable("crimes")

  # parse date/time
  strings = sqlContext.sql("SELECT Date, Arrest FROM crimes WHERE Arrest = true")
  timestamp = strings.withColumn("Date", unix_timestamp("Date", "MM/dd/yyyy hh:mm:ss a")\
                              .cast("double").cast("timestamp"))

  # Get Day of Week, Month, and Hour
  timestamp.registerTempTable("crimes")
  DOW = sqlContext.sql("SELECT date_format(Date, 'E') AS DOW, Arrest FROM crimes")
  Month = timestamp.withColumn("Month", month("Date")).select("Month", "Arrest")
  Hour = timestamp.withColumn("Hour", hour("Date")).select("Hour", "Arrest")

  # Roll up above data, plot, save plots
  Month = Month.toPandas()\
    .groupby("Month")\
    .count()\
    .apply(lambda x: x/x.sum())

  fig = Month.plot.bar(title = "Proportion of Arrests by Month ", rot = 0)
  fig.set_ylabel("Proportion of Arrests")
  final = fig.get_figure()
  final.savefig("exercise4_month.pdf")

  DOW = DOW.toPandas()\
    .groupby("DOW")\
    .count()\
    .apply(lambda x: x/x.sum())

  fig = DOW.plot.bar(title = "Proportion of Arrests by DOW ", rot = 0)
  fig.set_ylabel("Proportion of Arrests")
  final = fig.get_figure()
  final.savefig("exercise4_DOW.pdf")

  Hour = Hour\
    .toPandas()\
    .groupby("Hour")\
    .count()\
    .apply(lambda x: x/x.sum())
    
  fig = Hour.plot.bar(title = "Proportion of Arrests by Hour ", rot = 0)
  fig.set_ylabel("Proportion of Arrests")
  final = fig.get_figure()
  final.savefig("exercise4_hour.pdf")

  sc.stop()