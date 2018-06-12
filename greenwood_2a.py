from pyspark import SparkContext
from csv import reader

def parse_line(data):
  csv_reader = reader([data])
  fields = None
  for row in csv_reader:
      fields = row
  return fields


if __name__ == "__main__":
  # set up drivers
  sc = SparkContext(appName='Question 2')

  # read in data and get rid of header
  file = sc.textFile("hdfs://wolf.iems.northwestern.edu/user/sgreenwood/crime/Crimes_-_2001_to_present.csv")
  header = file.first()
  file = file.filter(lambda x: x != header)

  # subset by year, aggregate, and sort
  lines = file.map(parse_line)\
      .filter(lambda year: year[17] in ['2015', '2014', '2013'])\
      .map(lambda x: (x[3], 1))\
      .reduceByKey(lambda x, y: x + y)\
      .sortBy(lambda x: x[1], ascending=False)

  # find top 10 and write out
  result = lines.take(10)
  file = open('exercise2a.txt', 'w')
  file.write(str(result))
  file.close()