from pyspark.sql import DataFrame

def validate_data(df: DataFrame) -> DataFrame:
    # Example: drop rows with null values
    df_clean = df.dropna()
    return df_clean
