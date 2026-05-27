import os
import time
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import mlflow.transformers
from dotenv import load_dotenv
from tweet_sentiment_classifier import logger
from tweet_sentiment_classifier.config.config_entity import ModelEvaluationConfig
from tweet_sentiment_classifier.utils.common import save_json

load_dotenv()
import warnings
import logging
warnings.filterwarnings("ignore", message=".*artifact_path is deprecated.*")
logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        acc = accuracy_score(actual, pred)
        _, _, f1_w, _ = precision_recall_fscore_support(actual, pred, average='weighted', zero_division=0)
        _, _, f1_m, _ = precision_recall_fscore_support(actual, pred, average='macro', zero_division=0)
        return acc, f1_w, f1_m

    def log_into_mlflow(self):
        try:
            logger.info("Loading test dataset for model evaluation")
            test_df = pd.read_csv(self.config.test_data_path)
            
            X_test = test_df['text'].fillna("")
            y_test = test_df['sentiment']

            model_name = self.config.model_name
            hyperparams = self.config.hyperparams

            logger.info(f"Evaluating model: {model_name}")
            start_time = time.time()

            if model_name in ["LogisticRegression", "LinearSVC", "MultinomialNB", "RidgeClassifier", "XGBoost"]:
                model = joblib.load(self.config.model_path)
                vectorizer = joblib.load("artifacts/data_transformation/preprocessor.pkl")
                X_test_vectorized = vectorizer.transform(X_test)

                if model_name == "XGBoost":
                    pred_mapped = model.predict(X_test_vectorized)
                    predictions = pred_mapped - 1.0
                else:
                    predictions = model.predict(X_test_vectorized)

            elif model_name == "DistilBERT":
                import os as env_os
                env_os.environ["TF_USE_LEGACY_KERAS"] = "1"
                import tensorflow as tf
                from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

                sample_size = min(50, len(test_df))
                sampled_test_df = test_df.sample(n=sample_size, random_state=self.config.random_state)
                X_test = sampled_test_df['text']
                y_test = sampled_test_df['sentiment']

                tokenizer = AutoTokenizer.from_pretrained(str(self.config.model_path))
                model = TFAutoModelForSequenceClassification.from_pretrained(str(self.config.model_path))

                X_test_str = X_test.astype(str).tolist()
                test_encodings = tokenizer(X_test_str, truncation=True, padding=True, max_length=128, return_tensors="tf")

                logits = model(dict(test_encodings)).logits
                predictions = tf.argmax(logits, axis=1).numpy() - 1.0

            end_time = time.time()
            latency = end_time - start_time

            acc, f1_w, f1_m = self.eval_metrics(y_test, predictions)
            
            metrics = {
                "accuracy": float(acc),
                "f1_score_weighted": float(f1_w),
                "f1_score_macro": float(f1_m),
                "latency_seconds": float(latency)
            }

            metrics_dir = Path(self.config.metrics_file).parent
            metrics_file_path = metrics_dir / f"metrics_{model_name}.json"
            os.makedirs(metrics_file_path.parent, exist_ok=True)
            save_json(path=metrics_file_path, data=metrics)

            summary_path = metrics_dir / "all_models_summary.csv"
            new_row = {
                "Model Name": [model_name],
                "Accuracy": [acc],
                "F1 Score (Weighted)": [f1_w],
                "F1 Score (Macro)": [f1_m],
                "Latency": [latency]
            }
            new_df = pd.DataFrame(new_row)
            
            if summary_path.exists():
                summary_df = pd.read_csv(summary_path)
                summary_df = summary_df[summary_df["Model Name"] != model_name]
                summary_df = pd.concat([summary_df, new_df], ignore_index=True)
            else:
                summary_df = new_df
                
            summary_df.to_csv(summary_path, index=False)

            # Confusion Matrix
            cm = confusion_matrix(y_test, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Sentiment')
            plt.ylabel('Actual Sentiment')
            plt.title(f'Confusion Matrix - {model_name}')
            cm_plot_path = Path(self.config.root_dir) / f"confusion_matrix_{model_name}.png"
            plt.savefig(cm_plot_path, bbox_inches='tight')
            plt.close()

            # Report
            report = classification_report(y_test, predictions, zero_division=0)
            report_file_path = Path(self.config.root_dir) / f"classification_report_{model_name}.txt"
            with open(report_file_path, "w") as f:
                f.write(report)

            mlflow.set_experiment("Tweet_Sentiment_Classification")
            with mlflow.start_run(run_name=f"Run_{model_name}"):
                mlflow.log_params(hyperparams)
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(cm_plot_path), artifact_path="evaluation_outputs")
                mlflow.log_artifact(str(report_file_path), artifact_path="evaluation_outputs")
                
                if model_name in ["LogisticRegression", "LinearSVC", "MultinomialNB", "RidgeClassifier", "XGBoost"]:
                    preprocessor_path = Path("artifacts/data_transformation/preprocessor.pkl")
                    if preprocessor_path.exists():
                        mlflow.log_artifact(str(preprocessor_path), artifact_path="vectorizer")
                    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
                elif model_name == "DistilBERT":
                    mlflow.transformers.log_model(
                        transformers_model={"model": model, "tokenizer": tokenizer},
                        artifact_path="model"
                    )

            logger.info("MLflow tracking completed successfully")

        except Exception as e:
            logger.exception(f"Error during model evaluation: {e}")
            raise e
