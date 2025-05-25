"""sklearn-based model for hotel reservations prediction.

This module implements a basic model for predicting house prices using LightGBM.
It trains a model on a dataset, logs it with MLflow, and registers it in Unity Catalog.
"""

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger
from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hotel_reservations.config import ProjectConfig, Tags


class BasicModel:
    """A basic model class for house price prediction using LightGBM.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration.

        :param config: Project configuration object
        :param tags: Tags object
        :param spark: SparkSession object
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name
        self.model_name = f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model"
        self.tags = tags.dict()

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set_hotel_reservations")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.test_set_hotel_reservations"
        ).toPandas()
        self.data_version = "0"  # describe history -> retrieve

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self) -> None:
        """Encode categorical features and define a preprocessing pipeline.

        Creates a ColumnTransformer for one-hot encoding categorical features while passing through numerical
        features. Constructs a pipeline combining preprocessing and LightGBM regression model.
        """
        logger.info("🔄 Defining preprocessing pipeline...")
        self.preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("regressor", LGBMRegressor(**self.parameters))]
        )
        logger.info("✅ Preprocessing pipeline defined.")

    def finetune_parameters(self) -> None:
        """Fine-tune model hyperparameters."""
        logger.info("🔄 Fine-tuning model parameters...")
        param_grid = self.config.parameters

        # Convert to pipeline parameter format
        pipeline_param_grid = {}
        for key, value in param_grid.items():
            pipeline_param_grid[f"regressor__{key}"] = value

        mlflow.set_experiment(self.experiment_name)

        best_score = float("-inf")
        best_params = None
        best_run_id = None

        param_combinations = list(ParameterGrid(pipeline_param_grid))
        logger.info(f"🚀 Starting hyperparameter tuning with {len(param_combinations)} combinations...")

        with mlflow.start_run(tags=self.tags) as parent_run:
            self.run_id = parent_run.info.run_id
            for i, params in enumerate(param_combinations):
                # Create child run for each parameter combination
                with mlflow.start_run(run_name=f"run_{i + 1}", nested=True) as child_run:
                    try:
                        # Create pipeline with current parameters
                        current_pipeline = Pipeline(
                            [
                                ("preprocessor", self.preprocessor),
                                ("regressor", LGBMRegressor(random_state=42, verbose=-1)),
                            ]
                        )

                        # Set parameters
                        current_pipeline.set_params(**params)

                        # Train model
                        current_pipeline.fit(self.X_train, self.y_train)

                        # Make predictions
                        y_pred = current_pipeline.predict(self.X_test)

                        # Calculate metrics
                        mse = mean_squared_error(self.y_test, y_pred)
                        mae = mean_absolute_error(self.y_test, y_pred)
                        r2 = r2_score(self.y_test, y_pred)

                        # Log parameters (convert back to original format for clarity)
                        original_params = {k.replace("regressor__", ""): v for k, v in params.items()}
                        mlflow.log_params(original_params)

                        # Log metrics
                        mlflow.log_metric("mse", mse)
                        mlflow.log_metric("mae", mae)
                        mlflow.log_metric("r2_score", r2)

                        logger.info(f"📊 Run {i + 1}/{len(param_combinations)} - R2: {r2:.4f}, MSE: {mse:.4f}")

                        # Track best model
                        if r2 < best_score:  # Using R2 as the metric to minimize
                            best_score = r2
                            best_params = original_params
                            best_run_id = child_run.info.run_id

                    except Exception as e:
                        logger.error(f"❌ Error in run {i + 1}: {str(e)}")
                        mlflow.log_param("error", str(e))

            # Log best results in parent run
            mlflow.log_metric("best_r2_score", best_score)
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_param("best_run_id", best_run_id)

            logger.info(f"✅ Best R2 Score: {best_score:.4f}")
            logger.info(f"📊 Best Parameters: {best_params}")
            logger.info(f"🎯 Best Run ID: {best_run_id}")

            # Store best model info for registration
            self.best_run_id = best_run_id
            self.best_params = best_params
            self.best_score = best_score

            # Log model
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set_hotel_reservations",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")
            mlflow.sklearn.log_model(
                sk_model=current_pipeline, artifact_path="lightgbm-pipeline-model", signature=signature
            )

    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version,
        )

    def retrieve_current_run_dataset(self) -> DatasetSource:
        """Retrieve MLflow run dataset.

        :return: Loaded dataset source
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        logger.info("✅ Dataset source loaded.")
        return dataset_source.load()

    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """Retrieve MLflow run metadata.

        :return: Tuple containing metrics and parameters dictionaries
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("✅ Dataset metadata loaded.")
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Load the latest model from MLflow (alias=latest-model) and make predictions.

        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param input_data: Pandas DataFrame containing input features for prediction.
        :return: Pandas DataFrame with predictions.
        """
        logger.info("🔄 Loading model from MLflow alias 'production'...")

        model_uri = f"models:/{self.model_name}@latest-model"
        model = mlflow.sklearn.load_model(model_uri)

        logger.info("✅ Model successfully loaded.")

        # Make predictions
        predictions = model.predict(input_data)

        # Return predictions as a DataFrame
        return predictions
