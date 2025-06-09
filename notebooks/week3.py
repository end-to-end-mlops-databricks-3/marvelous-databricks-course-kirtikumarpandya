# Databricks notebook source
# %pip install -e ..
# %pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0

# COMMAND ----------

# %restart_python

# COMMAND ----------

import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent / "src"))

# COMMAND ----------

# Configure tracking uri
from databricks.sdk.runtime import display
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.feature_lookup_model import FeatureLookUpModel

# Configure tracking uri
# mlflow.set_tracking_uri("databricks")
# mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "week2"}
tags = Tags(**tags_dict)

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------

fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

fe_model.create_feature_table()

# COMMAND ----------

fe_model.define_feature_functions()

# COMMAND ----------

fe_model.load_data()

# COMMAND ----------

fe_model.feature_engineering()

# COMMAND ----------

fe_model.train_model()

# COMMAND ----------

fe_model.register_model()

# COMMAND ----------
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)
X_test = f"{config.catalog_name}.{config.schema_name}.test_set_hotel_reservations"
X_test = spark.table(X_test)
X_test = X_test.withColumn(
    "market_segment_type", expr("mlops_dev.1potdish.convert_market_segment_type(market_segment_type)")
)
X_test = X_test.withColumn("type_of_meal_plan", expr("mlops_dev.1potdish.convert_type_of_meal_plan(type_of_meal_plan)"))
X_test = X_test.withColumn(
    "room_type_reserved", expr("mlops_dev.1potdish.convert_room_type_plan(room_type_reserved)")
)  # .toPandas()
X_test = X_test.select(*fe_model.lookup_features, *config.id_cols)

predictions = fe_model.load_latest_model_and_predict(X_test)

# COMMAND ----------

display(predictions)
