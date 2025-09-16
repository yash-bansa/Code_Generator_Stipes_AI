from pyspark.sql import DataFrame
from src.utils.spark_session import get_spark_session

def read_csv(file_path: str) -> DataFrame:
    spark = get_spark_session()
    df = spark.read.option("header", True).csv(file_path)
    return df
