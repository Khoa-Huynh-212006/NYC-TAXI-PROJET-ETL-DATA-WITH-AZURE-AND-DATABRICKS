# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze -> Silver

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load dữ liệu -> Table

# COMMAND ----------

spark.table('workspace.project_nyc_taxi.df_green_2025_01_08').display()

# COMMAND ----------

import pyspark
from pyspark.sql import * 
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import *
spark = SparkSession.builder.appName('Nyc_taxi').getOrCreate()


# COMMAND ----------

df = spark.read.format('parquet')\
                .option('header', 'true')\
                .option('inferSchema', 'true')\
                .load('workspace.project_nyc_taxi.df_green_2025_01_06')

# COMMAND ----------

df_2024_01_06 = spark.table('workspace.project_nyc_taxi.df_green_2025_01_06')
df_2024_07_12 = spark.table('workspace.project_nyc_taxi.df_green_2024_07_12')
df_2025_01_08 = spark.table('workspace.project_nyc_taxi.df_green_2025_01_08')
df_trip_type = spark.table('workspace.project_nyc_taxi.trip_type')
df_trip_zone = spark.table('workspace.project_nyc_taxi.trip_zone')

# COMMAND ----------

df_2024_01_06.limit(5).display()

# COMMAND ----------

df_2024_07_12.printSchema()

# COMMAND ----------

df_trip_type.display()
df_trip_zone.limit(5).display()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Union Table 

# COMMAND ----------

df_2025_01_08.limit(5).display()

# COMMAND ----------

df_2024 = df_2024_01_06.unionByName(df_2024_07_12)
df_2024.withColumn('cbd_congestion_fee', lit(0)).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Xử lý tiền dữ liệu

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Hiểu dữ liệu

# COMMAND ----------

df_2024.printSchema()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Cast Data type

# COMMAND ----------

df_2024.createTempView('df_2024')

# COMMAND ----------

spark.sql(""" 
          SELECT * FROM df_2024""").limit(5).display()

# COMMAND ----------

df_2024 = spark.sql(""" 
              SELECT CAST(VendorID AS INT) AS VendorID,
                     CAST(lpep_pickup_datetime AS TIMESTAMP) AS lpep_pickup_datetime,
                     CAST(lpep_dropoff_datetime AS TIMESTAMP) AS lpep_dropoff_datetime,
                     CAST(store_and_fwd_flag AS STRING) AS store_and_fwd_flag,
                     CAST(RatecodeID AS INT) AS RatecodeID,
                     CAST(PULocationID AS INT) AS PULocationID,
                     CAST(DOLocationID AS INT) AS DOLocationID,
                     CAST(passenger_count AS INT) AS passenger_count,
                     CAST(trip_distance AS DOUBLE) AS trip_distance,
                     CAST(fare_amount AS DOUBLE) AS fare_amount,
                     CAST(extra AS DOUBLE) AS extra,
                     CAST(mta_tax AS DOUBLE) AS mta_tax,
                     CAST(tip_amount AS DOUBLE) AS tip_amount,
                     CAST(tolls_amount AS DOUBLE) AS tolls_amount,
                     CAST(improvement_surcharge AS DOUBLE) AS improvement_surcharge,
                     CAST(total_amount AS DOUBLE) AS total_amount,
                     CAST(payment_type AS INT) AS payment_type,
                     CAST(trip_type AS INT) AS trip_type,
                     CAST(congestion_surcharge AS DOUBLE) AS congestion_surcharge
                    FROM df_2024
              """)

# COMMAND ----------

df_2024.limit(5).display()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Làm sạch/ Chuẩn hóa/ Biến đổi dữ liệu

# COMMAND ----------

import pandas as pd 
df_pd = df_2024.toPandas()
df_pd.info()

# COMMAND ----------

# MAGIC %md
# MAGIC store_and_fwd_flag
# MAGIC
# MAGIC RatecodeID    
# MAGIC
# MAGIC passenger_count   
# MAGIC
# MAGIC payment_type
# MAGIC
# MAGIC congestion_surcharge
# MAGIC
# MAGIC có cùng 1 giá trị là 843543
# MAGIC -> các giá missing làm cho dữ liệu không rõ ràng -> xóa 
# MAGIC
# MAGIC trip_type có 843438
# MAGIC -> các giá trị missing vẫn có thể hiểu được dựa trên data còn lại -> giữ và chuyển về unknown, 0
# MAGIC
# MAGIC ehail_fee full null values -> cột rác

# COMMAND ----------

df_pd.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Null/Missing Values

# COMMAND ----------

df_2024 = df_2024.drop('ehail_fee')
df_2024.limit(5).display()

# COMMAND ----------

df_2024_handle_null = df_2024.na.drop(how = 'any', subset = ['store_and_fwd_flag', 'RatecodeID', 'passenger_count', 'payment_type', 'congestion_surcharge'])
df_2024_handle_null.filter(df_2024['congestion_surcharge'].isNull()).display()

# COMMAND ----------

df_trip_type.display()

# COMMAND ----------

df_2024_handle_null = df_2024_handle_null.join(df_trip_type, df_2024_handle_null['trip_type'] == df_trip_type['trip_type'], "left").drop(df_trip_type['trip_type'])

df_2024_handle_null = df_2024_handle_null.na.fill({'trip_type': 0, 'description': 'Unknown'})


# COMMAND ----------

df_pd=  df_2024_handle_null.toPandas()
df_pd.info()


# COMMAND ----------

# MAGIC %md 
# MAGIC #### Duplicate Values

# COMMAND ----------

df_2024_handle_null.selectExpr(
    "count(distinct VendorID) as dis_VendorID",
    "count(distinct RatecodeID) as dis_RatecodeID",
    "count(distinct PULocationID) as dis_PULocationID",
    "count(distinct DOLocationID) as dis_DOLocationID",
    "count(distinct passenger_count) as dis_passenger_count",
    "count(distinct trip_distance) as dis_trip_distance",
    "count(distinct fare_amount) as dis_fare_amount",
    "count(distinct extra) as dis_extra",
    "count(distinct mta_tax) as dis_mta_tax",
    "count(distinct tip_amount) as dis_tip_amount",
    "count(distinct tolls_amount) as dis_tolls_amount",
    "count(distinct improvement_surcharge) as dis_improvement_surcharge",
    "count(distinct total_amount) as dis_total_amount",
    "count(distinct payment_type) as dis_payment_type",
    "count(distinct trip_type) as dis_trip_type",
    "count(distinct congestion_surcharge) as dis_congestion_surcharge",
    "count(distinct store_and_fwd_flag) as dis_store_and_fwd_flag",
    "count(distinct description) as dis_description",
    "count(distinct lpep_pickup_datetime) as dis_lpep_pickup_datetime",
    "count(distinct lpep_dropoff_datetime) as dis_lpep_dropoff_datetime"
).display()


# COMMAND ----------

df_2024_handle_null.limit(5).display()

# COMMAND ----------

df_trip_zone.display()


# COMMAND ----------

df_trip_zone.display()

# COMMAND ----------

df_trip_zone = df_trip_zone.withColumn('Zone1', split(col('Zone'), '/')[0])\
            .withColumn('Zone2', when(size(split(col('Zone'), '/')) > 1, split(col('Zone'), '/')[1]).otherwise(None))

# COMMAND ----------

df_trip_zone = df_trip_zone.drop('Zone')

# COMMAND ----------

df_joined = df_2024_handle_null.join(df_trip_zone, df_2024_handle_null['PULocationID'] == df_trip_zone['LocationID'], "left").withColumnRenamed('Borough', 'PUBorough').withColumnRenamed('Zone1', 'PUZone1').withColumnRenamed('Zone2', 'PUZone2').withColumnRenamed('service_zone', 'PUServiceZone').drop('LocationID')



df_joined = df_joined.join(df_trip_zone, df_2024_handle_null['DOLocationID'] == df_trip_zone['LocationID'], "left").withColumnRenamed('Borough', 'DOBorough').withColumnRenamed('Zone1', 'DOZone1').withColumnRenamed('Zone2', 'DOZone2').withColumnRenamed('service_zone', 'DOServiceZone').drop('LocationID')
df_joined.display()

# COMMAND ----------

df_duplicated = df_joined.dropDuplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Outliers

# COMMAND ----------

df_trans = df_duplicated.withColumn('Day', dayofmonth(col('lpep_pickup_datetime')))\
            .withColumn('Month', month(col('lpep_pickup_datetime')))\
            .withColumn('Year', year(col('lpep_pickup_datetime')))\

df_pd = df_trans.toPandas()

df_pd.describe()

# COMMAND ----------

df_trans.createTempView('df_sql')

# COMMAND ----------

# MAGIC %md
# MAGIC trip_distance: max lên tới 35024 dặm -> quá lớn so với trung bình là 2.7 -> lớn hơn 50 -> outlier
# MAGIC
# MAGIC fare_amount: min -450 ? -> < 0 là giá trị rác
# MAGIC
# MAGIC

# COMMAND ----------

spark.sql("SELECT * FROM df_sql WHERE trip_distance > 50").display()

# COMMAND ----------

df_trans = df_trans.filter(df_trans['trip_distance'] <= 50)

# COMMAND ----------

spark.sql("SELECT * FROM df_sql WHERE fare_amount > 100").display()

# COMMAND ----------

df_trans = df_trans.filter((df_trans['fare_amount'] <= 100) & (df_trans['fare_amount'] > 0))

# COMMAND ----------

df_pd = df_trans.toPandas()
df_pd.describe()

# COMMAND ----------

spark.sql("SELECT * FROM df_sql WHERE tip_amount > 30").display()

# COMMAND ----------

df_trans = df_trans.filter(df_trans['tip_amount'] <= 30)

# COMMAND ----------

spark.sql("SELECT * FROM df_sql WHERE tolls_amount > 10").display()

# COMMAND ----------

df_trans = df_trans.filter(df_trans['tolls_amount'] <= 10)

# COMMAND ----------

df_pd = df_trans.toPandas()
df_pd.describe()


# COMMAND ----------

df_trans.limit(5).display()

# COMMAND ----------

rename_list = {
               'lpep_pickup_datetime' : 'PickupDateTime',
               'lpep_dropoff_datetime': 'DropoffDateTime',
               'store_and_fwd_flag': 'StoreAndFwdFlag',
               'passenger_count': 'PassengerCount',
               'trip_distance': 'TripDistance',
               'RatecodeID': 'RateCodeID',
               'fare_amount': 'FareAmount',
               'extra': 'Extra',
               'mta_tax': 'MTATax',
               'tip_amount': 'TipAmount',
               'tolls_amount': 'TollsAmount',
               'improvement_surcharge': 'ImprovementSurcharge',
               'total_amount': 'TotalAmount',
               'payment_type': 'PaymentType',
               'trip_type': 'TripType',
               'congestion_surcharge': 'CongestionSurcharge',
               'description': 'Description'
                }

# COMMAND ----------

for i in rename_list:
  df_trans = df_trans.withColumnRenamed(i, rename_list[i])

df_trans.limit(5).display()


# COMMAND ----------

df_trans.display()

# COMMAND ----------

#thêm cột số giờ pickup tới dropoff
df_trans = df_trans.withColumn("DurationMinutes",((unix_timestamp(col("DropoffDateTime")) - unix_timestamp(col("PickupDateTime"))) / 60))


# COMMAND ----------

df_trans.display()

# COMMAND ----------

#làm tròn cột DurationMinutes
df_trans = df_trans.withColumn("DurationMinutes", round(col("DurationMinutes"), 0))
df_trans.display()

# COMMAND ----------

df_trans.write.mode('overwrite').saveAsTable('df_trans')

# COMMAND ----------

df_trans.display()


# COMMAND ----------

df_trans = df_trans.drop('PickupDateTime','DropoffDateTime')

# COMMAND ----------

df_trans.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Silver -> Gold 
# MAGIC

# COMMAND ----------

df_silver = spark.table('workspace.project_nyc_taxi.project_nyc_taxi_silver')

# COMMAND ----------

df_silver.limit(5).display()

# COMMAND ----------

df_silver.write.format('delta').mode('overwrite').saveAsTable('gold_nyc_taxi')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM gold_nyc_taxi

# COMMAND ----------

df_type = spark.table('workspace.project_nyc_taxi.trip_type')
df_zone = spark.table('workspace.project_nyc_taxi.trip_zone')

df_type.write.format('delta').mode('overwrite').saveAsTable('gold_trip_type')
df_zone.write.format('delta').mode('overwrite').saveAsTable('gold_trip_zone')