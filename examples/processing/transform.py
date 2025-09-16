from pyspark.sql import DataFrame
from pyspark.sql.functions import col

def transform_data(df: DataFrame) -> DataFrame:
    # Example: Convert column to integer
    df_transformed = df.withColumn("value", col("value").cast("int"))
    return df_transformed
