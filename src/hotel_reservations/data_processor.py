import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from hotel_reservations.config import ProjectConfig
from hotel_reservations.helper import rename_spark_df_column_name


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
        pandas_df = pandas_df.drop_duplicates()
        pandas_df = pandas_df.dropna(how="all")

        # drop target column with None values and reset index
        pandas_df = pandas_df.dropna(subset=[self.config.target])
        pandas_df = pandas_df.reset_index(drop=True)

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
            + pandas_df["arrival_date"].astype(str).str.zfill(2),
            dayfirst=False,
            format="%Y-%m-%d",
            errors="coerce",
        )
        pandas_df = pandas_df.dropna(subset=["date"]).reset_index(drop=True)

        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            pandas_df[col] = pd.to_numeric(pandas_df[col], errors="coerce")

        self.pandas_df = pandas_df[
            self.config.num_features + self.config.cat_features + self.config.id_cols + [self.config.target]
        ]

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set, test_set = train_test_split(self.pandas_df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp = rename_spark_df_column_name(train_set_with_timestamp)
        test_set_with_timestamp = rename_spark_df_column_name(test_set_with_timestamp)

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set_hotel_reservations"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set_hotel_reservations"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set_hotel_reservations"
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set_hotel_reservations "
            " SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
