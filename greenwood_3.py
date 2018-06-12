from pyspark import SparkContext
import pandas as pd
from csv import reader
from datetime import datetime
from dateutil.parser import parse
from pyspark.sql.functions import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.window import Window
import datetime
from pyspark.sql import HiveContext
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import OneHotEncoder, StringIndexer


if __name__ == "__main__":
  # set up drivers
  sc = SparkContext()
  sqlContext = HiveContext(sc)

  # read in data
  file = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true') \
   .load('hdfs://wolf.iems.northwestern.edu/user/sgreenwood/crime/Crimes_-_2001_to_present.csv')
  file.registerTempTable("crimes")

  # transform to timestamp
  strings = sqlContext.sql("SELECT Date, Beat, Ward FROM crimes")
  timestamp = strings.withColumn("Date", unix_timestamp("Date", "MM/dd/yyyy hh:mm:ss a")\
                              .cast("double").cast("timestamp"))

  # Convert Timestamp to Month/Week/Year
  Month = timestamp.withColumn("Month", month("Date"))
  Week = Month.withColumn("Week", weekofyear("Date"))
  Year = Week.withColumn("Year", year("Date"))

  # Aggreagate crimes per timeframe and beat
  final = Year.groupBy("Month", "Week", "Year","Beat").count()

  # Generate Beat/Ward Key for additional dataset and grouped dataset
  key = timestamp.select("Beat", "Ward").distinct()

  # Get ward back in grouping!
  final = final.join(key, final.Beat == key.Beat)\
    .select("Month", "Week", "Year", key.Beat, "count", "Ward")\
    .withColumnRenamed("count", "Crimes")

  # Read in new dataset
  file2 = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true')\
    .load("hdfs://wolf.iems.northwestern.edu/user/sgreenwood/crime/Ordinance_Violations.csv")
  file2.registerTempTable("extra")

  # Manipulate new dataset: Violations per Ward
  extra = sqlContext.sql("SELECT WARD, `VIOLATION DATE` FROM extra")
  timestamp = extra.withColumn("Date", unix_timestamp("VIOLATION DATE", "MM/dd/yyyy hh:mm:ss a")\
                              .cast("double").cast("timestamp"))
  Month = timestamp.withColumn("Month_extra", month("Date"))
  Week = Month.withColumn("Week_extra", weekofyear("Date"))
  Year = Week.withColumn("Year_extra", year("Date"))
  extra = Year.groupBy("Month_extra", "Week_extra", "Year_extra", "WARD")\
    .count()\
    .select("Month_extra", "Week_extra", "Year_extra", "WARD", "count")\
    .withColumnRenamed("count", "Violations")

  # Merge original and new dataset
  cond = [final.Month == extra.Month_extra, final.Week == extra.Week_extra, final.Year == extra.Year_extra, final.Ward == extra.WARD]
  final_join = final.join(extra, cond).select("Month", "Week", "Beat", "Year", "Crimes", "Violations")

  # Lag Violations 
  window = Window.orderBy("Beat", "Year", "Month", "Week")
  lag_df = final_join.withColumn("Lag_Violations", lag("Violations", 1, 0).over(window))
  lag_df = lag_df.withColumn("Crimes", lag_df["Crimes"].cast(DoubleType()))

  # Index Categorical Variables (Beat, Year, Month, Week)
  beatIndexer = StringIndexer(inputCol="Beat", outputCol="beatIndex")
  model = beatIndexer.fit(lag_df)
  lag_df = model.transform(lag_df)

  yearIndexer = StringIndexer(inputCol="Year", outputCol="yearIndex")
  model = yearIndexer.fit(lag_df)
  lag_df = model.transform(lag_df)

  monthIndexer = StringIndexer(inputCol="Beat", outputCol="monthIndex")
  model = monthIndexer.fit(lag_df)
  lag_df = model.transform(lag_df)

  # One Hot Encode Categorical Variables
  encoder = OneHotEncoder(dropLast=False, inputCol="beatIndex", outputCol="beatVec")
  lag_df = encoder.transform(lag_df)

  encoder = OneHotEncoder(dropLast=False, inputCol="yearIndex", outputCol="yearVec")
  lag_df = encoder.transform(lag_df)

  encoder = OneHotEncoder(dropLast=False, inputCol="monthIndex", outputCol="monthVec")
  lag_df = encoder.transform(lag_df)

  # Make train/test split with last week in dataset
  test = lag_df.filter((lag_df["Year"] == 2015) & (lag_df["Week"] == 21))
  train = lag_df.filter((lag_df["Year"] != 2015) & (lag_df["Week"] != 21))

  # Vectorize train and test
  vectorAssembler = VectorAssembler(inputCols = ['monthVec', 'beatVec', 'yearVec', 'Lag_Violations'], outputCol = 'features')
  train = vectorAssembler.transform(train)
  train = train.select(['features', 'Crimes'])
  test = vectorAssembler.transform(test)
  test = test.select(['features', 'Crimes'])

  # Fit Model
  lr = LinearRegression(featuresCol = 'features', labelCol='Crimes', maxIter=10)
  lr_model = lr.fit(train)

  # Write out coefficients
  coef = "Coefficients: " + str(lr_model.coefficients)
  intercept = "Intercept: " + str(lr_model.intercept)

  text_file = open("exercise3.txt", "w")
  text_file.write(coef)
  text_file.write(intercept)

  # Predict for last timeframe in dataset (test set)
  lr_predictions = lr_model.transform(test)
  lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="Crimes",metricName="rmse")

  # Save results to CSV
  RMSE = str("RMSE: " + str(lr_evaluator.evaluate(lr_predictions)))
  text_file.write(RMSE)
  results = lr_predictions.toPandas()
  results.to_csv("exercise3.csv", index = False)

  # Turn off driver
  sc.stop()
