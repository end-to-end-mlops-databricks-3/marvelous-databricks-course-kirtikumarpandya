# Databricks notebook source

# % pip install -e ..
# %restart_python

import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent / "src"))

# COMMAND ----------

import sys

import pandas as pd
import yaml
from loguru import logger
from marvelous.logging import setup_logging
from marvelous.timer import Timer
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

setup_logging(log_file="logs/hotel_reservations-1.log")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))
# COMMAND ----------
# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()

filepath = "../data/data.csv"

# Load the data
df = pd.read_csv(filepath)

# COMMAND ----------
with Timer() as preprocess_timer:
    # Initialize DataProcessor
    data_processor = DataProcessor(config=config, pandas_df=df, spark=spark)

    # Preprocess the data
    data_processor.preprocess()

logger.info(f"Data preprocessing: {preprocess_timer}")
# COMMAND ----------
# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)
# COMMAND ----------
X_train.head()
# COMMAND ----------
# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

# Enable change data feed (only once!)
# logger.info("Enable change data feed")
# data_processor.enable_change_data_feed()
# COMMAND ----------
logger.info("Setting primary and foreign keys")
data_processor.set_primary_and_foreign_keys()
# COMMAND ----------
