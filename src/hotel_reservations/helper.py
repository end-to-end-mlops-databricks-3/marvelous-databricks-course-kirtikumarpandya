from pyspark.sql import DataFrame


def rename_spark_df_column_name(spark_df: DataFrame) -> DataFrame:
    """Rename columns in a Spark DataFrame to ensure compatibility with SQL queries.

    This method replaces spaces, parentheses, curly braces, commas, semicolons,
    and equal signs in column names with underscores or brackets.

    :param spark_df: Spark DataFrame with columns to be renamed
    :return: Spark DataFrame with renamed columns
    """
    spark_df = spark_df.toDF(*[col.replace(" ", "_") for col in spark_df.columns])
    spark_df = spark_df.toDF(*[col.replace("(", "[") for col in spark_df.columns])
    spark_df = spark_df.toDF(*[col.replace(")", "]") for col in spark_df.columns])
    spark_df = spark_df.toDF(*[col.replace("{", "[") for col in spark_df.columns])
    spark_df = spark_df.toDF(*[col.replace("}", "]") for col in spark_df.columns])
    spark_df = spark_df.toDF(*[col.replace(",", "_") for col in spark_df.columns])
    spark_df = spark_df.toDF(*[col.replace(";", "_") for col in spark_df.columns])
    spark_df = spark_df.toDF(*[col.replace("=", "_") for col in spark_df.columns])

    return spark_df
