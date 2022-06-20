from pyspark.sql import SparkSession
from pyspark.sql.functions import *
#from pyspark.sql.functions import split, regexp_extract 


spark = SparkSession.builder \
        .master("local[2]") \
        .appName("Lab 7 Exercise") \
        .config("spark.local.dir","/fastdata/acr21aa") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 


logFile = spark.read.text("../Data/NASA_access_log_Jul95.gz").cache() 
logFile.show(20, False)

# Splitting the log values into different columns for future use
df = logFile.select(regexp_extract('value', r'^([^\s]+\s)', 1).alias('host_ip'),regexp_extract('value', r'^.*\[(\d\d/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]', 1).alias('timestamp'),regexp_extract('value', r'^.*"\w+\s+([^\s]+)\s+HTTP.*"', 1).alias('request_path'),regexp_extract('value', r'^.*"\s+([^\s]+)', 1).cast('integer').alias('status'),regexp_extract('value', r'^.*\s+(\d+)$', 1).cast('integer').alias('content_size'))

df.show(1, False)

# Converting the timestamp into spark default format 
# so as to extract weekday from it
df = df.withColumn('timestamp' , to_timestamp('timestamp', 'dd/MMM/yyyy:HH:mm:ss -SSSS'))
df = df.withColumn('weekday', dayofweek('timestamp'))
df = df.withColumn('date', dayofmonth('timestamp'))

# finding the number of request on each weekday 
df2 = df.groupBy('weekday', 'date').count()
df2 = df2.orderBy('weekday')

min_max_req = df2.groupBy('weekday').agg(min('count').alias('minimum_requests'), max('count').alias('maximum_requests'))
min_max_req.orderBy('weekday').show()

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

f, (ax1, ax2) = plt.subplots(2, 1)

# Converting dataframe columns into lists to use in plot
weekdays = min_max_req.select('weekday').rdd.flatMap(lambda x: x).collect() 
min_req = min_max_req.select('minimum_requests').rdd.flatMap(lambda x: x).collect()
max_req = min_max_req.select('maximum_requests').rdd.flatMap(lambda x: x).collect()


ax1.plot(weekdays, min_req, marker='o')
ax1.set_title('Minimum no. of requests each day')
ax2.plot(weekdays, max_req, marker='o')
ax2.set_title('Maximum no. of requests each day')
plt.savefig("../Output/min_max_req.png")


#filter mpg videos from request path
mpg_videos=df.filter(df.request_path.contains('.mpg')).groupBy('request_path').count().orderBy('count')
mpg_videos.show(50, False)

# list of most and least requested videos
most_and_least = mpg_videos.take(12)
most_and_least = most_and_least + mpg_videos.tail(12)

df4 = spark.createDataFrame(most_and_least)
df4.show(24, False)

f = plt.figure()
path = df4.select('request_path').rdd.flatMap(lambda x: x).collect()
count = df4.select('count').rdd.flatMap(lambda x: x).collect()
plt.bar(path, count)
plt.xticks(rotation='vertical')
plt.xlabel("Requested videos path")
plt.ylabel("No. of requests")
plt.title("Most and Least Requested Videos")
plt.savefig("../Output/most_least_req.png")


# get the names of most and least requested videos
df4 = df4.withColumn('name', split('request_path', '/'))
df4.select('request_path', 'count',element_at('name', -1).alias('video_name')).show(24, False)