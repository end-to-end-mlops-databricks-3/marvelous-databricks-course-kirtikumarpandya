import pandas as pd
import yaml
from loguru import logger
from marvelous.common import create_parser
from marvelous.logging import setup_logging
from pyspark.sql import SparkSession

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))
from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor

args = create_parser()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
is_test = args.is_test

# config_path = Path.cwd() / "project_config.yml"
# config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

setup_logging(log_file=f"/Volumes/{config.catalog_name}/{config.schema_name}/logs/hotel_reservations-1.log")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()
filepath = "../data/data.csv"

# Load the data
df = pd.read_csv(filepath)

# Preprocess the data
data_processor = DataProcessor(pandas_df=df, config=config, spark=spark)
data_processor.preprocess()

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)
