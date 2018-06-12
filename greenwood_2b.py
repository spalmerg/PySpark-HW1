from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.stat import Statistics
import pandas as pd
from csv import reader

# fill in dates
def fill(beats_on_date, all_beats):
  final = []
  for beat in all_beats: 
      value = 0
      if(beat in beats_on_date):
          value = beats_on_date.count(beat)
      final.append(value)
  return final

# funtion to help read in and account for commas in description
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
      .map(parse_line)\
      .filter(lambda year: year[17] in ['2015', '2014', '2013', '2012', '2011'])\
      .map(lambda x: ((x[2][0:2] + x[2][5:10]), x[10]))
  
  # identify all beats
  beats = lines.map(lambda x: x[1])\
      .distinct().collect()
  
  # key = beats, values = list of crime month/year
  unfilled = lines.reduceByKey(lambda x, y: x + "," + y)\
      .map(lambda x: (x[0], x[1].split(",")))

  # count number of crimes per day per beat, fill no-crime values with zero
  filled = unfilled.map(lambda x: (x[0], fill(x[1], beats)))

  # convert to vectors
  vectors = filled.map(lambda x: Vectors.dense(x[1]))

  # calculate correlation
  pearsonCorr = Statistics.corr(vectors)

  # identify top 30 correlated beats
  pearsonCorr = pd.DataFrame(pearsonCorr, index = beats, columns = beats)
  unstacked = pearsonCorr.unstack()
  unstacked = pd.DataFrame(unstacked).reset_index()
  unstacked.columns = ["beat1", "beat2", "correlation"]
  unstacked = unstacked[unstacked.beat1 != unstacked.beat2]
  final = unstacked.nlargest(300, "correlation")

  # write final to csv
  final.to_csv("greenwood_2b.csv", index=False)

  # stop driver
  sc.stop()