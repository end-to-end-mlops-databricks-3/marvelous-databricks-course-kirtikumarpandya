# Databricks notebook source
import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from lightgbm import LGBMClassifier

sys.path.append(str(Path.cwd().parent / "src"))

from marvelous.common import is_databricks
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hotel_reservations.config import ProjectConfig

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
train_df = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set_hotel_reservations").toPandas()
test_df = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set_hotel_reservations").toPandas()
# COMMAND ----------
X_train = train_df[config.num_features + config.cat_features + config.date_features]
X_test = test_df[config.num_features + config.cat_features + config.date_features]
y_train = train_df[config.target]
y_test = test_df[config.target]
# COMMAND ----------
features_preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), config.cat_features)], remainder="passthrough"
)

classifier = LGBMClassifier()
pipeline = Pipeline(steps=[("features_preprocessor", features_preprocessor), ("classifier", classifier)])

pipeline.fit(X_train, y_train)
# COMMAND ----------
if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

# COMMAND ----------
print(mlflow.get_tracking_uri())
print(mlflow.get_registry_uri())
# COMMAND ----------
