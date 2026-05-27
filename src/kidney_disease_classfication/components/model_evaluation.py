import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
from kidney_disease_classfication import logger
from kidney_disease_classfication.config.config_entity import ModelEvaluationConfig
from kidney_disease_classfication.utils.common import save_json

# Load environment variables from .env file
load_dotenv()

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        """
        Calculates accuracy, precision, recall, and F1-score.
        """
        acc = accuracy_score(actual, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(actual, pred, average='weighted', zero_division=0)
        return acc, precision, recall, f1

    def log_into_mlflow(self):
        """
        Evaluates the chosen trained model on the test dataset and logs parameters,
        metrics, and the model artifact to MLflow/DagsHub, while saving metrics locally.
        Uses lazy imports and sampling constraints for high-speed, lightweight resource usage.
        """
        try:
            logger.info("Loading test dataset for model evaluation")
            test_df = pd.read_csv(self.config.test_data_path)
            test_df['cleaned_text_final'] = test_df['cleaned_text_final'].fillna("")

            # Set up features and target
            X_test = test_df['cleaned_text_final']
            y_test = test_df['sentiment']

            model_name = self.config.model_name
            logger.info(f"Evaluating model: {model_name}")

            if model_name in ["LogisticRegression", "MultinomialNB"]:
                logger.info(f"Loading trained classifier model from {self.config.model_path}")
                model = joblib.load(self.config.model_path)

                logger.info("Loading preprocessor (TF-IDF vectorizer) for text transformation")
                vectorizer = joblib.load("artifacts/data_transformation/preprocessor.pkl")

                # Transform test features
                X_test_vectorized = vectorizer.transform(X_test)

                # Make predictions
                predictions = model.predict(X_test_vectorized)

            elif model_name == "GlobalPoolingFFN":
                # Lazy import TensorFlow
                logger.info("Lazy importing TensorFlow for FFN model evaluation")
                import tensorflow as tf

                logger.info(f"Loading Keras FFN model from {self.config.model_path}")
                model = tf.keras.models.load_model(str(self.config.model_path))

                # Make predictions
                X_test_str = X_test.astype(str).tolist()
                logger.info("Generating predictions on test set via Keras")
                prob_preds = model.predict(X_test_str, batch_size=128)
                
                # Map back from [0, 1, 2] classes to [-1.0, 0.0, 1.0] sentiments
                predictions = np.argmax(prob_preds, axis=1) - 1.0

            elif model_name == "DistilBERT":
                # Lazy import TensorFlow and Transformers
                logger.info("Lazy importing TensorFlow and Transformers for DistilBERT evaluation")
                import tensorflow as tf
                from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

                # Sample test set to 50 records for ultra-fast, lightweight evaluation under CPU limitations
                sample_size = min(50, len(test_df))
                logger.info(f"Sampling {sample_size} records to evaluate DistilBERT safely under hardware constraints")
                sampled_test_df = test_df.sample(n=sample_size, random_state=42)
                X_test = sampled_test_df['cleaned_text_final']
                y_test = sampled_test_df['sentiment']

                logger.info(f"Loading tokenizer and TF DistilBERT model from {self.config.model_path}")
                tokenizer = AutoTokenizer.from_pretrained(str(self.config.model_path))
                model = TFAutoModelForSequenceClassification.from_pretrained(str(self.config.model_path))

                # Tokenize inputs
                X_test_str = X_test.astype(str).tolist()
                test_encodings = tokenizer(X_test_str, truncation=True, padding=True, max_length=128, return_tensors="tf")

                # Make predictions
                logger.info("Generating predictions on test set via DistilBERT")
                logits = model(dict(test_encodings)).logits
                predictions = tf.argmax(logits, axis=1).numpy() - 1.0

            else:
                raise ValueError(f"Invalid model name '{model_name}' specified in params.yaml!")

            # Compute evaluation metrics
            acc, precision, recall, f1 = self.eval_metrics(y_test, predictions)
            metrics = {
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }
            logger.info(f"Evaluation metrics computed: {metrics}")

            # Save metrics locally as JSON
            metrics_file_path = Path(self.config.metrics_file)
            os.makedirs(metrics_file_path.parent, exist_ok=True)
            save_json(path=metrics_file_path, data=metrics)
            logger.info(f"Saved evaluation metrics locally at {metrics_file_path}")

            # MLflow/DagsHub Tracking Setup
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
            tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

            # Use local MLflow runs if tracking credentials are not set/placeholder
            if tracking_uri and "YOUR_DAGSHUB_TOKEN_OR_PASSWORD" not in tracking_password:
                logger.info(f"Connecting to remote MLflow server at: {tracking_uri}")
                mlflow.set_tracking_uri(tracking_uri)
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            else:
                logger.warning(
                    "MLflow remote tracking credentials not configured or placeholder detected in .env! "
                    "Logging metrics locally in mlruns directory instead. Fill in credentials in .env to connect to DagsHub."
                )
                tracking_url_type_store = "file"

            # Set mlflow experiment name
            mlflow.set_experiment("Kidney_Disease_Sentiment_Classification")

            with mlflow.start_run(run_name=f"Run_{model_name}"):
                logger.info("Starting MLflow tracking run")
                
                # Log hyperparameters & configurations
                mlflow.log_params({
                    "model_name": model_name,
                    "threshold": self.config.threshold,
                    "stage": self.config.stage
                })

                # Log metrics
                mlflow.log_metrics(metrics)

                # Log Model Artifacts
                if model_name in ["LogisticRegression", "MultinomialNB"]:
                    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
                else:
                    # Log Keras/Transformers models as generic artifacts
                    logger.info("Logging deep learning model directory as artifacts")
                    mlflow.log_artifacts(str(self.config.model_path), artifact_path="model")

            logger.info("MLflow tracking completed successfully")

        except Exception as e:
            logger.exception(f"Error encountered during model evaluation: {e}")
            raise e
