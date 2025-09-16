from pyspark.sql import SparkSession

def get_spark_session(app_name="DataPipelineApp"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .getOrCreate()
    return spark
