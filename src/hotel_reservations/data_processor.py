import pandas as pd
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig


class DataProcessor:
    """DataProcessor class for handling data processing tasks.

    This class is responsible for loading, transforming, and saving data.
    It uses PySpark for distributed data processing and Pandas for local operations.
    """

    def __init__(self, config: ProjectConfig, pandas_df: pd.DataFrame, spark: SparkSession) -> None:
        """Initialize the DataProcessor with a configuration object.

        :param config: ProjectConfig object containing configuration settings
        :param pandas_df: Pandas DataFrame to be processed
        :param spark: SparkSession object for distributed data processing
        """
        self.config = config
        self.pandas_df = pandas_df
        self.spark = spark

    def process(self) -> None:
        """Process the data by loading, transforming, and saving it.

        This method performs the following steps:
        1. Load data from a Pandas DataFrame.
        2. Transform the data using PySpark.
        3. Save the transformed data to a specified location.
        """
        pandas_df = self.pandas_df

        # drop duplicates and null values
        pandas_df = pandas_df.drop_duplicates().dropna(how="all")

        # drop target column with None values and reset index
        pandas_df = pandas_df.dropna(subset=[self.config.target]).reset_index(drop=True)

        # Converting id columns to string
        id_cols = self.config.id_cols
        for id_col in id_cols:
            pandas_df[id_col] = pandas_df[id_col].astype("str")

        # Convert categorical features to the appropriate type
        cat_features = self.config.cat_features
        for cat_col in cat_features:
            pandas_df[cat_col] = pandas_df[cat_col].astype("category")

        for date_feature in self.config.date_features:
            pandas_df[date_feature] = pandas_df[date_feature].astype("int")
            most_frequent_value = pandas_df[date_feature].mode()[0]
            # Fill missing values with the most frequent value
            pandas_df[date_feature].fillna(most_frequent_value, inplace=True)

        pandas_df["date"] = pd.to_datetime(
            pandas_df["arrival_year"].astype(str)
            + "-"
            + pandas_df["arrival_month"].astype(str).str.zfill(2)
            + "-"
            + pandas_df["arrival_date"].astype(str).str.zfill(2)
        )
