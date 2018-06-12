from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors, Matrices
from pyspark.mllib.stat import Statistics
import pandas as pd
from csv import reader
from datetime import datetime
from dateutil.parser import parse
import sys


def parse_line(data):
  csv_reader = reader([data])
  fields = None
  for row in csv_reader:
      fields = row
  return fields

if __name__ == "__main__":
  # set up driver
  sc = SparkContext(appName='Question 2')

  # read in data and get rid of header
  lines = sc.textFile('hdfs://wolf.iems.northwestern.edu/user/sgreenwood/crime/Crimes_-_2001_to_present.csv')
  header = lines.first()
  lines = lines.filter(lambda x: x != header)\
      .map(parse_line)

  # get Daly datset
  daly = lines.filter(lambda x: datetime.strptime(x[2][0:10], '%m/%d/%Y') < datetime.strptime("5/16/2011", '%m/%d/%Y'))
  daly_date = len(daly.map(lambda x: (x[2][0:2] + x[2][5:10])).distinct().collect())
  daly_final = daly.map(lambda x: (x[10][0:2], x[2][0:2] + x[2][5:10]))\
    .map(lambda x: (x,1))\
    .reduceByKey(lambda x, y: x+y)\
    .map(lambda x: (x[0][0], x[1]))\
    .reduceByKey(lambda x, y: (x+y))\
    .map(lambda x: (x[0], x[1]/daly_date))\
    .sortBy(lambda x: x[0])\
    .map(lambda x: x[1])

  # get Emanuel dataset
  emanuel = lines.filter(lambda x: datetime.strptime(x[2][0:10], '%m/%d/%Y') >= datetime.strptime("5/16/2011", '%m/%d/%Y'))
  emanuel_date = len(emanuel.map(lambda x: (x[2][0:2] + x[2][5:10])).distinct().collect())
  emanuel_final = emanuel.map(lambda x: (x[10][0:2], x[2][0:2] + x[2][5:10]))\
    .map(lambda x: (x,1))\
    .reduceByKey(lambda x, y: x+y)\
    .map(lambda x: (x[0][0], x[1]))\
    .reduceByKey(lambda x, y: (x+y))\
    .map(lambda x: (x[0], x[1]/emanuel_date))\
    .sortBy(lambda x: x[0])\
    .map(lambda x: x[1])

  # Convert to vector for Pyspark Chi Squared Formatting
  daly_vec = Vectors.dense(daly_final.collect())
  emanuel_vec = Vectors.dense(emanuel_final.collect())

  # Calculate Chi Squared Stat
  pearson = Statistics.chiSqTest(daly_vec, emanuel_vec)
  output = str(pearson)

  text_file = open("exercise2c.txt", "w")
  text_file.write(output)

  sc.stop()

