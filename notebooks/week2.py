# Databricks notebook source
# % pip install -e ..
# %restart_python
# COMMAND ----------
import sys
from pathlib import Path

import mlflow

sys.path.append(str(Path.cwd().parent / "src"))

from databricks.sdk.runtime import spark

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.model import BasicModel

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
# COMMAND ----------
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
tags = Tags(git_sha="dsadsdsds323rf", branch="main")
model = BasicModel(config, tags, spark)
# COMMAND ----------
model.load_data()
model.prepare_features()
model.finetune_parameters()
model.register_model()

# COMMAND ----------
