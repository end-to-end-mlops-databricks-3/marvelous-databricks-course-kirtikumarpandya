import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import spark
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import expr
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hotel_reservations.config import ProjectConfig, Tags


class FeatureLookUpModel:
    """FeatureLookupModel class for handling feature lookup operations.

    This class is responsible for creating and managing feature lookups in Databricks.
    It uses the Feature Engineering client to create and manage feature tables.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the FeatureLookupModel with a configuration object.

        :param config: ProjectConfig object containing configuration settings
        :param tags: Tags object containing metadata for the model
        :param spark: SparkSession object for distributed data processing
        """
        self.config = config
        self.tags = tags
        self.spark = spark
        self.fe_client = feature_engineering.FeatureEngineeringClient()
        self.workspace_client = WorkspaceClient()

        # Extract features from config
        self.num_features = config.num_features
        self.cat_features = config.cat_features
        self.date_features = config.date_features
        self.id_cols = config.id_cols
        self.target = config.target
        self.parameters = config.parameters
        self.catalog_name = config.catalog_name
        self.schema_name = config.schema_name
        self.experiment_name = config.experiment_name_fe
        self.tags = tags.dict()

        self.lookup_features = self.num_features + self.cat_features + self.date_features

        self.model_name = f"{self.catalog_name}.{self.schema_name}.hotel_reservations_fe_model"
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.hotel_reservations_features"

    def create_feature_table(self) -> None:
        """Create the feature table in Databricks."""
        self.fe_client.create_table(
            name=self.feature_table_name,
            primary_keys=self.id_cols,
            df=self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set_hotel_reservations")[
                self.lookup_features + self.id_cols
            ],
            description="Feature table for hotel reservations.",
        )

        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.fe_client.write_table(
            name=self.feature_table_name,
            df=self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set_hotel_reservations")[
                self.lookup_features + self.id_cols
            ],
            mode="merge",
        )

        logger.info("✅ Feature table created and populated.")

    def define_feature_functions(self) -> None:
        """Define feature functions for the feature lookup."""
        self.function_name_meal = f"{self.catalog_name}.{self.schema_name}.convert_type_of_meal_plan"
        self.spark.sql(f"""
                CREATE OR REPLACE FUNCTION {self.function_name_meal}(type_of_meal_plan STRING)
                RETURNS INT
                LANGUAGE PYTHON AS
                $$
                match type_of_meal_plan:
                    case "Meal Plan 1": return 1
                    case "Meal Plan 2": return 2
                    case "Meal Plan 3": return 3
                    case _: return 0
                $$
                """)

        self.function_name_room_type = f"{self.catalog_name}.{self.schema_name}.convert_room_type_plan"
        self.spark.sql(f"""
                CREATE OR REPLACE FUNCTION {self.function_name_room_type}(room_type_plan STRING)
                RETURNS INT
                LANGUAGE PYTHON AS
                $$
                match room_type_plan:
                    case "Room_Type 1": return 1
                    case "Room_Type 2": return 2
                    case "Room_Type 3": return 3
                    case "Room_Type 4": return 4
                    case "Room_Type 5": return 5
                    case "Room_Type 6": return 6
                    case "Room_Type 7": return 7
                    case _: return 0
                $$
                """)

        self.function_name_market_segment = f"{self.catalog_name}.{self.schema_name}.convert_market_segment_type"
        self.spark.sql(f"""
                CREATE OR REPLACE FUNCTION {self.function_name_market_segment}(market_segment_type STRING)
                RETURNS INT
                LANGUAGE PYTHON AS
                $$
                match market_segment_type:
                    case "Complementary": return 1
                    case "Aviation": return 2
                    case "Corporate": return 3
                    case "Online": return 4
                    case "Offline": return 5
                    case _: return 0
                $$
                """)
        logger.info("✅ Feature functions defined.")

    def load_data(self) -> None:
        """Load training and test datasets from the catalog."""
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set_hotel_reservations").drop(
            "type_of_meal_plan", "room_type_reserved", "market_segment_type"
        )
        self.train_set = self.train_set.withColumn(self.id_cols[0], self.train_set[self.id_cols[0]].cast("string"))

        self.test_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.test_set_hotel_reservations"
        ).toPandas()
        logger.info("✅ Data loaded successfully.")

    def feature_engineering(self) -> None:
        """Perform feature engineering on the training and test sets."""
        self.training_set = self.fe_client.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    lookup_key=self.config.id_cols[0],
                    feature_names=["type_of_meal_plan", "room_type_reserved", "market_segment_type"],
                    rename_outputs={
                        "type_of_meal_plan": "meal_plan",
                        "room_type_reserved": "room_type",
                        "market_segment_type": "market_segment",
                    },
                ),
                FeatureFunction(
                    udf_name=self.function_name_market_segment,
                    output_name="market_segment_type",
                    input_bindings={"market_segment_type": "market_segment"},
                ),
                FeatureFunction(
                    udf_name=self.function_name_meal,
                    output_name="type_of_meal_plan",
                    input_bindings={"type_of_meal_plan": "meal_plan"},
                ),
                FeatureFunction(
                    udf_name=self.function_name_room_type,
                    output_name="room_type_reserved",
                    input_bindings={"room_type_plan": "room_type"},
                ),
            ],
            exclude_columns=["update_timestamp_utc", "date", "meal_plan", "room_type", "market_segment"],
        )
        self.training_df = self.training_set.load_df().toPandas()
        self.X_train = self.training_df[self.lookup_features]
        self.y_train = self.training_df[self.target]
        self.X_test = spark.createDataFrame(self.test_set[self.lookup_features])
        self.X_test = self.X_test.withColumn(
            "market_segment_type", expr(f'{self.function_name_market_segment}("market_segment_type")')
        )
        self.X_test = self.X_test.withColumn(
            "type_of_meal_plan", expr(f'{self.function_name_meal}("type_of_meal_plan")')
        )
        self.X_test = self.X_test.withColumn(
            "room_type_reserved", expr(f'{self.function_name_room_type}("room_type_reserved")')
        ).toPandas()
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train_model(self) -> None:
        """Fine-tune model hyperparameters."""
        logger.info("🔄 Fine-tuning model parameters...")
        param_grid = self.config.parameters

        # Convert to pipeline parameter format
        pipeline_param_grid = {}
        for key, value in param_grid.items():
            pipeline_param_grid[f"classifier__{key}"] = value

        mlflow.set_experiment(self.experiment_name)

        best_accuracy = float("-inf")
        best_params = None
        best_run_id = None

        param_combinations = list(ParameterGrid(pipeline_param_grid))
        logger.info(f"🚀 Starting hyperparameter tuning with {len(param_combinations)} combinations...")

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier())])

        with mlflow.start_run(tags=self.tags) as parent_run:
            self.run_id = parent_run.info.run_id
            for i, params in enumerate(param_combinations):
                # Create child run for each parameter combination
                with mlflow.start_run(run_name=f"run_{i + 1}", nested=True) as child_run:
                    try:
                        # Set parameters
                        pipeline.set_params(**params)

                        # Train model
                        pipeline.fit(self.X_train, self.y_train)

                        # Make predictions
                        y_pred = pipeline.predict(self.X_test)

                        # Calculate metrics
                        accuracy = accuracy_score(self.y_test, y_pred)

                        # Log parameters (convert back to original format for clarity)
                        original_params = {k.replace("classifier__", ""): v for k, v in params.items()}
                        mlflow.log_params(original_params)

                        # Log metrics
                        mlflow.log_metric("accuracy", accuracy)

                        logger.info(f"📊 Run {i + 1}/{len(param_combinations)} - Accuracy: {accuracy:.4f}")

                        if best_params is None:
                            best_params = original_params

                        # Track best model
                        if accuracy > best_accuracy:  # Using accuracy as the metric to minimize
                            best_accuracy = accuracy
                            best_params = original_params
                            best_run_id = child_run.info.run_id
                            self.best_pipeline = pipeline

                    except Exception as e:
                        logger.error(f"❌ Error in run {i + 1}: {str(e)}")
                        mlflow.log_param("error", str(e))

            # Log best results in parent run
            mlflow.log_metric("best_accuracy", best_accuracy)
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_param("best_run_id", best_run_id)

            logger.info(f"✅ Best Accuracy: {best_accuracy:.4f}")
            logger.info(f"📊 Best Parameters: {best_params}")
            logger.info(f"🎯 Best Run ID: {best_run_id}")

            # Store best model info for registration
            self.best_run_id = best_run_id
            self.best_params = best_params
            self.best_accuracy = best_accuracy

            # Log model
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)
            self.fe_client.log_model(
                model=self.best_pipeline,
                flavor=mlflow.sklearn,
                training_set=self.training_set,
                signature=signature,
                artifact_path="lightgbm-pipeline-model-fe",
            )

    def register_model(self) -> None:
        """Register the trained model to MLflow registry.

        Registers the model and sets alias to 'latest-model'.
        """
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.best_run_id}/lightgbm-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe",
            alias="latest-model",
            version=latest_version,
        )

        return latest_version

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Loads the model with the alias 'latest-model' and scores the batch.
        :param X: DataFrame containing the input features.
        :return: DataFrame containing the predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe@latest-model"

        predictions = self.fe_client.score_batch(model_uri=model_uri, df=X)
        return predictions
