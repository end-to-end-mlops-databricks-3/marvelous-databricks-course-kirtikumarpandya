# Databricks notebook source
# MAGIC %pip install -e ..
# MAGIC %pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent / "src"))

# COMMAND ----------

import os
import time

import pandas as pd
import requests
from databricks.sdk.runtime import display
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig
from hotel_reservations.serving.model_serving import ModelServing

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.hotel_reservations_model",
    endpoint_name="hotel-reservations-model-serving",
)

# COMMAND ----------

model_serving.deploy_or_update_serving_endpoint()

# COMMAND ----------

# Sample 1000 records from the training set
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set_hotel_reservations").toPandas()

# COMMAND ----------
required_columns = config.cat_features + config.num_features + config.date_features
sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------


def call_endpoint(record: dict) -> tuple:
    """Call the model serving endpoint with a given input record."""
    serving_endpoint = (
        f"https://{os.environ['DBR_HOST']}/serving-endpoints/hotel-reservations-model-serving/invocations"
    )

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


# COMMAND ----------

status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------

Response_status = []
Response_text = []

# Load test
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    assert status_code == 200
    Response_status.append(status_code)
    Response_text.append(eval(response_text)["predictions"][0])
    time.sleep(0.2)

# COMMAND ----------
predictions = pd.DataFrame()
predictions["status_code"] = Response_status
predictions["response_text"] = Response_text
map_response_text = spark.table(f"{catalog_name}.{schema_name}.target_mapping_hotel_reservations").toPandas()
predictions["predicted"] = predictions["response_text"].map({0: "Not_Canceled", 1: "Canceled"})
display(predictions)

# COMMAND ----------
