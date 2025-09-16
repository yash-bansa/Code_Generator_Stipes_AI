from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler

class FeatureEngineering:
    def __init__(self):
        pass

    def add_features(df: DataFrame, feature_cols: list):
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df_features = assembler.transform(df)
        return df_features
