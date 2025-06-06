"""Unit tests for DataProcessor."""

import pandas as pd
import pytest
from conftest import CATALOG_DIR
from delta.tables import DeltaTable
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor
from hotel_reservations.helper import rename_spark_df_column_name


def test_data_ingestion(sample_data: pd.DataFrame) -> None:
    """Test the data ingestion process by checking the shape of the sample data.

    Asserts that the sample data has at least one row and one column.

    :param sample_data: The sample data to be tested
    """
    assert sample_data.shape[0] > 0
    assert sample_data.shape[1] > 0


def test_dataprocessor_init(
    sample_data: pd.DataFrame,
    config: ProjectConfig,
    spark_session: SparkSession,
) -> None:
    """Test the initialization of DataProcessor.

    :param sample_data: Sample DataFrame for testing
    :param config: Configuration object for the project
    :param spark: SparkSession object
    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    assert isinstance(processor.pandas_df, pd.DataFrame)
    assert processor.pandas_df.equals(sample_data)

    assert isinstance(processor.config, ProjectConfig)
    assert isinstance(processor.spark, SparkSession)


def test_column_transformations(sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test the column transformations in the DataProcessor.

    This function checks if the columns are transformed correctly based on the configuration.
    :param sample_data: Input DataFrame containing sample data
    :param config: Configuration object for the project
    :param spark: SparkSession object
    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()

    assert "GarageYrBlt" not in processor.pandas_df.columns
    assert processor.pandas_df["Booking_ID"].dtype == "object"
    assert processor.pandas_df["type_of_meal_plan"].dtype == "category"


def test_column_selection(sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test column selection in the DataProcessor.

    This function checks if the correct columns are selected and present in the
    processed DataFrame based on the configuration.

    :param sample_data: Input DataFrame containing sample data
    :param config: Configuration object for the project
    :param spark: SparkSession object
    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()

    expected_columns = (
        config.cat_features + config.num_features + [config.target, "Booking_ID", "date"] + config.date_features
    )
    assert set(processor.pandas_df.columns) == set(expected_columns)


def test_split_data_default_params(
    sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test the default parameters of the split_data method in DataProcessor.

    This function tests if the split_data method correctly splits the input DataFrame
    into train and test sets using default parameters.

    :param sample_data: Input DataFrame to be split
    :param config: Configuration object for the project
    :param spark: SparkSession object
    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()
    train, test = processor.split_data()

    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) + len(test) == len(processor.pandas_df)
    assert set(train.columns) == set(test.columns) == set(processor.pandas_df.columns)

    # # The following lines are just to mimick the behavior of delta tables in UC
    # # Just one time execution in order for all other tests to work
    train.to_csv((CATALOG_DIR / "train_set.csv").as_posix(), index=False)  # noqa
    test.to_csv((CATALOG_DIR / "test_set.csv").as_posix(), index=False)  # noqa


def test_preprocess_empty_dataframe(config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test the preprocess method with an empty DataFrame.

    This function tests if the preprocess method correctly handles an empty DataFrame
    and raises KeyError.

    :param config: Configuration object for the project
    :param spark: SparkSession object
    :raises KeyError: If the preprocess method handles empty DataFrame correctly
    """
    processor = DataProcessor(pandas_df=pd.DataFrame([]), config=config, spark=spark_session)
    with pytest.raises(KeyError):
        processor.preprocess()


@pytest.mark.skip(reason="depends on delta tables on Databricks")
def test_save_to_catalog_succesfull(
    sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test the successful saving of data to the catalog.

    This function processes sample data, splits it into train and test sets, and saves them to the catalog.
    It then asserts that the saved tables exist in the catalog.

    :param sample_data: The sample data to be processed and saved
    :param config: Configuration object for the project
    :param spark: SparkSession object for interacting with Spark
    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()
    train_set, test_set = processor.split_data()
    processor.save_to_catalog(train_set, test_set)
    processor.enable_change_data_feed()

    # Assert
    assert spark_session.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.train_set_hotel_reservations")
    assert spark_session.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.test_set_hotel_reservations")


@pytest.mark.skip(reason="depends on delta tables on Databrics")
@pytest.mark.order(after=test_save_to_catalog_succesfull)
def test_delta_table_property_of_enableChangeDataFeed_check(config: ProjectConfig, spark_session: SparkSession) -> None:
    """Check if Change Data Feed is enabled for train and test sets.

    Verifies that the 'delta.enableChangeDataFeed' property is set to True for both
    the train and test set Delta tables.

    :param config: Project configuration object
    :param spark: SparkSession object
    """
    train_set_path = f"{config.catalog_name}.{config.schema_name}.train_set_hotel_reservations"
    test_set_path = f"{config.catalog_name}.{config.schema_name}.test_set_hotel_reservations"
    tables = [train_set_path, test_set_path]
    for table in tables:
        delta_table = DeltaTable.forName(spark_session, table)
        properties = delta_table.detail().select("properties").collect()[0][0]
        cdf_enabled = properties.get("delta.enableChangeDataFeed")
        assert bool(cdf_enabled) is True


@pytest.mark.skip(reason="depends on delta tables on Databricks")
@pytest.mark.order(after=test_delta_table_property_of_enableChangeDataFeed_check)
def test_set_primary_and_foreign_keys(config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test the primary and foreign key settings of the DataProcessor.

    This function tests if the primary and foreign keys are correctly set for the train and test sets.

    :param config: Configuration object for the project
    :param spark: SparkSession object
    """
    processor = DataProcessor(pandas_df=pd.DataFrame([]), config=config, spark=spark_session)
    processor.set_primary_and_foreign_keys()

    assert spark_session.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.train_set_hotel_reservations")
    assert spark_session.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.test_set_hotel_reservations")


def test_rename_spark_df_column_name(spark_df_with_special_chars: SparkSession) -> None:
    """Test that rename_spark_df_column_name correctly replaces special characters in column names.

    :param spark_df_with_special_chars: Spark DataFrame with special character column names
    """
    # Apply the function to rename columns
    renamed_df = rename_spark_df_column_name(spark_df_with_special_chars)

    # Get the new column names
    new_columns = renamed_df.columns

    # Check each column name was transformed correctly
    assert "column_with_space" in new_columns
    assert "column[with]parentheses" in new_columns
    assert "column[with]braces" in new_columns
    assert "column_with_commas" in new_columns
    assert "column_with_semicolons" in new_columns
    assert "column_with_equals" in new_columns
    assert "mixed_column[with][all]_special_chars_together" in new_columns
    assert "normal_column" in new_columns  # This should remain unchanged

    # Verify that no original special characters remain in any column name
    for col in new_columns:
        assert " " not in col, f"Space found in column name: {col}"
        assert "(" not in col, f"Opening parenthesis found in column name: {col}"
        assert ")" not in col, f"Closing parenthesis found in column name: {col}"
        assert "{" not in col, f"Opening brace found in column name: {col}"
        assert "}" not in col, f"Closing brace found in column name: {col}"
        assert "," not in col, f"Comma found in column name: {col}"
        assert ";" not in col, f"Semicolon found in column name: {col}"
        assert "=" not in col, f"Equal sign found in column name: {col}"
