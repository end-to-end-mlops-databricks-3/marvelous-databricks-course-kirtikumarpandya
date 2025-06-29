# Databricks notebook source
# % pip install -e ..
# %restart_python
# COMMAND ----------

# import sys
# from pathlib import Path
# sys.path.append(str(Path.cwd().parent / "src"))
from databricks.sdk.runtime import dbutils, spark
from loguru import logger
from marvelous.common import create_parser

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.model import BasicModel

args = create_parser()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
# COMMAND ----------
tags = Tags(git_sha=args.git_sha, branch=args.branch)
model = BasicModel(config, tags, spark)
logger.info("Model initialized with configuration: %s", config)
# COMMAND ----------
model.load_data()
model.prepare_features()
model.finetune_parameters()

# COMMAND ----------
is_test = args.is_test
if is_test == 0:
    model_improved = model.model_improved()
    if model_improved:
        logger.info("Model has improved, registering the model.")
        model.register_model()
        dbutils.jobs.taskValues.set(key="model_updated", value=1)
    else:
        dbutils.jobs.taskValues.set(key="model_updated", value=0)
        logger.info("Model has not improved, skipping registration.")
else:
    logger.info("Running in test mode, registering the model regardless of improvement status.")
