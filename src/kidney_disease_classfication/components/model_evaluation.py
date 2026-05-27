import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.transformers
from kidney_disease_classfication import logger
from kidney_disease_classfication.config.config_entity import ModelEvaluationConfig
from kidney_disease_classfication.utils.common import save_json

# Load environment variables from .env file
load_dotenv()

import warnings
warnings.filterwarnings("ignore", message=".*artifact_path is deprecated.*")

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        """
        Calculates accuracy, weighted and macro precision, recall, and F1-score.
        """
        acc = accuracy_score(actual, pred)
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(actual, pred, average='weighted', zero_division=0)
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(actual, pred, average='macro', zero_division=0)
        return acc, precision_w, recall_w, f1_w, precision_m, recall_m, f1_m

    def log_into_mlflow(self):
        """
        Evaluates the chosen trained model on the test dataset and logs parameters,
        metrics, plots, and model artifacts to MLflow/DagsHub, while saving metrics locally.
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
            hyperparams = self.config.hyperparams
            logger.info(f"Evaluating model: {model_name}")

            if model_name in ["LogisticRegression", "SVM", "MultinomialNB", "RidgeClassifier", "XGBoost"]:
                logger.info(f"Loading trained classifier model from {self.config.model_path}")
                model = joblib.load(self.config.model_path)

                logger.info("Loading preprocessor (TF-IDF vectorizer) for text transformation")
                vectorizer = joblib.load("artifacts/data_transformation/preprocessor.pkl")

                # Transform test features
                X_test_vectorized = vectorizer.transform(X_test)

                # Make predictions
                if model_name == "XGBoost":
                    pred_mapped = model.predict(X_test_vectorized)
                    # Map back from [0, 1, 2] classes to [-1.0, 0.0, 1.0] sentiments
                    predictions = pred_mapped - 1.0
                else:
                    predictions = model.predict(X_test_vectorized)

            elif model_name == "GlobalPoolingFFN":
                # Lazy import TensorFlow
                logger.info("Lazy importing TensorFlow for FFN model evaluation")
                import tensorflow as tf

                keras_model_path = str(self.config.model_path).replace(".pkl", ".keras")
                logger.info(f"Loading Keras FFN model from {keras_model_path}")
                model = tf.keras.models.load_model(keras_model_path)

                # Make predictions
                X_test_raw = X_test.astype(str).tolist()
                X_test_str = np.array([s.encode('cp1252', errors='ignore').decode('cp1252') for s in X_test_raw], dtype=object)
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
                sampled_test_df = test_df.sample(n=sample_size, random_state=self.config.random_state)
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
            acc, prec_w, rec_w, f1_w, prec_m, rec_m, f1_m = self.eval_metrics(y_test, predictions)
            metrics = {
                "accuracy": float(acc),
                "precision_weighted": float(prec_w),
                "recall_weighted": float(rec_w),
                "f1_score_weighted": float(f1_w),
                "precision_macro": float(prec_m),
                "recall_macro": float(rec_m),
                "f1_score_macro": float(f1_m)
            }
            logger.info(f"Evaluation metrics computed: {metrics}")

            # Save metrics locally as a model-specific JSON file to prevent overwriting
            metrics_dir = Path(self.config.metrics_file).parent
            metrics_file_path = metrics_dir / f"metrics_{model_name}.json"
            os.makedirs(metrics_file_path.parent, exist_ok=True)
            save_json(path=metrics_file_path, data=metrics)
            logger.info(f"Saved evaluation metrics locally at {metrics_file_path}")

            # Update a consolidated summary ledger for all trained models locally
            summary_path = metrics_dir / "all_models_summary.csv"
            columns_to_keep = ["Timestamp", "Model Name", "Accuracy", "F1 Score (Weighted)", "F1 Score (Macro)", "Precision (Weighted)", "Recall (Weighted)"]
            
            new_row = {
                "Timestamp": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
                "Model Name": [model_name],
                "Accuracy": [acc],
                "F1 Score (Weighted)": [f1_w],
                "F1 Score (Macro)": [f1_m],
                "Precision (Weighted)": [prec_w],
                "Recall (Weighted)": [rec_w]
            }
            new_df = pd.DataFrame(new_row)
            
            if summary_path.exists():
                try:
                    summary_df = pd.read_csv(summary_path)
                    # Filter existing to keep only clean columns
                    summary_df = summary_df[[c for c in columns_to_keep if c in summary_df.columns]]
                    summary_df = summary_df[summary_df["Model Name"] != model_name]
                    summary_df = pd.concat([summary_df, new_df], ignore_index=True)
                except Exception:
                    summary_df = new_df
            else:
                summary_df = new_df
                
            # Enforce exact column ordering and drop empty ones
            summary_df = summary_df.reindex(columns=columns_to_keep)
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Updated all models summary ledger at {summary_path}")

            # Create Confusion Matrix Plot
            cm = confusion_matrix(y_test, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[-1, 0, 1], yticklabels=[-1, 0, 1])
            plt.xlabel('Predicted Sentiment')
            plt.ylabel('Actual Sentiment')
            plt.title(f'Confusion Matrix - {model_name}')
            cm_plot_path = Path(self.config.root_dir) / f"confusion_matrix_{model_name}.png"
            os.makedirs(cm_plot_path.parent, exist_ok=True)
            plt.savefig(cm_plot_path, bbox_inches='tight')
            plt.close()

            # Create Text Classification Report
            report = classification_report(y_test, predictions, zero_division=0)
            report_file_path = Path(self.config.root_dir) / f"classification_report_{model_name}.txt"
            with open(report_file_path, "w") as f:
                f.write(report)

            # Determine if this model is the best performing model (highest accuracy) in the summary ledger
            is_best = False
            try:
                if summary_path.exists():
                    summary_df_check = pd.read_csv(summary_path)
                    max_acc = summary_df_check["Accuracy"].max()
                    # If this model's accuracy is equal to or greater than the max accuracy in the ledger, it is the best model!
                    if acc >= max_acc:
                        is_best = True
                        logger.info(f"Model {model_name} is currently the BEST model in the ledger with accuracy: {acc:.6f} (max was {max_acc:.6f})")
                else:
                    is_best = True
            except Exception as e:
                logger.warning(f"Error while checking for best model in summary ledger: {e}")

            registered_name = "Best_Sentiment_Model" if is_best else None
            if is_best:
                logger.info(f"Model {model_name} will be registered as '{registered_name}' in the Model Registry.")

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
                    "Logging metrics locally in mlruns directory instead."
                )
                tracking_url_type_store = "file"

            # Set mlflow experiment name
            mlflow.set_experiment("Kidney_Disease_Sentiment_Classification")

            with mlflow.start_run(run_name=f"Run_{model_name}"):
                logger.info(f"Starting MLflow tracking run: Run_{model_name}")
                
                # Log Common Hyperparameters
                mlflow.log_params({
                    "model_name": model_name,
                    "preprocessing_strategy": "TF-IDF Vectorization" if model_name not in ["GlobalPoolingFFN", "DistilBERT"] else "Text Tokenization",
                    "vectorizer_or_tokenizer": "TfidfVectorizer" if model_name not in ["GlobalPoolingFFN", "DistilBERT"] else ("Keras TextVectorization" if model_name == "GlobalPoolingFFN" else "DistilBertTokenizer"),
                    "train_test_split": "80-20",
                    "random_state": self.config.random_state
                })

                # Log Model-Specific Hyperparameters dynamically
                if model_name in ["LogisticRegression", "SVM", "MultinomialNB", "RidgeClassifier", "XGBoost"] and hasattr(model, "get_params"):
                    try:
                        actual_params = model.get_params()
                        tuned_params = {}
                        for k in hyperparams.keys():
                            if k in actual_params:
                                tuned_params[k] = actual_params[k]
                        if not tuned_params:
                            tuned_params = hyperparams
                        mlflow.log_params(tuned_params)
                    except Exception as e:
                        logger.warning(f"Could not extract tuned params from best model: {e}")
                        mlflow.log_params(hyperparams)
                else:
                    mlflow.log_params(hyperparams)

                # Log Classification Metrics
                mlflow.log_metrics(metrics)

                # Log Plots and Reports as Artifacts
                mlflow.log_artifact(str(cm_plot_path), artifact_path="evaluation_outputs")
                mlflow.log_artifact(str(report_file_path), artifact_path="evaluation_outputs")

                # Log Preprocessors and Models
                if model_name in ["LogisticRegression", "SVM", "MultinomialNB", "RidgeClassifier", "XGBoost"]:
                    preprocessor_path = Path("artifacts/data_transformation/preprocessor.pkl")
                    if preprocessor_path.exists():
                        mlflow.log_artifact(str(preprocessor_path), artifact_path="vectorizer")
                    mlflow.sklearn.log_model(sk_model=model, artifact_path="model", registered_model_name=registered_name)
                elif model_name == "GlobalPoolingFFN":
                    mlflow.tensorflow.log_model(model, artifact_path="model", registered_model_name=registered_name)
                elif model_name == "DistilBERT":
                    mlflow.transformers.log_model(
                        transformers_model={
                            "model": model,
                            "tokenizer": tokenizer
                        },
                        artifact_path="model",
                        registered_model_name=registered_name
                    )

            logger.info("MLflow tracking completed successfully")

        except Exception as e:
            logger.exception(f"Error encountered during model evaluation: {e}")
            raise e
