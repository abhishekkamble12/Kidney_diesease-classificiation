import os
import shutil
import pandas as pd
from pathlib import Path
from tweet_sentiment_classifier import logger
from tweet_sentiment_classifier.config.config_entity import BestModelConfig
from tweet_sentiment_classifier.utils.common import save_json

class BestModelSelector:
    def __init__(self, config: BestModelConfig):
        self.config = config

    def select_best_model(self):
        try:
            summary_path = Path(self.config.metrics_dir) / "all_models_summary.csv"
            if not summary_path.exists():
                logger.error(f"Summary file not found at {summary_path}")
                return False

            logger.info("Selecting best model based on macro_f1 > weighted_f1 > latency > accuracy")
            df = pd.read_csv(summary_path)

            # Sort by priority
            # Ascending for latency (lower is better), descending for metrics
            df_sorted = df.sort_values(
                by=['F1 Score (Macro)', 'F1 Score (Weighted)', 'Latency', 'Accuracy'],
                ascending=[False, False, True, False]
            )

            best_model_name = df_sorted.iloc[0]['Model Name']
            logger.info(f"Best model selected: {best_model_name}")

            # Persist best model artifact
            if best_model_name == "DistilBERT":
                best_model_src = Path("artifacts/model_trainer") / f"model_{best_model_name}"
            else:
                best_model_src = Path("artifacts/model_trainer") / f"model_{best_model_name}.pkl"
            
            dest_dir = Path(self.config.model_path).parent
            if best_model_src.is_dir():
                dest_path = dest_dir / f"best_model_{best_model_name}"
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(best_model_src, dest_path)
            else:
                dest_path = self.config.model_path
                shutil.copy2(best_model_src, dest_path)
            
            # Persist metadata
            metadata = {
                "best_model_name": best_model_name,
                "metrics": df_sorted.iloc[0].to_dict(),
                "tier_1_fast_fallback": df_sorted[df_sorted['Model Name'] != "DistilBERT"].iloc[0]['Model Name']
            }
            save_json(path=Path(self.config.metadata_path), data=metadata)
            logger.info(f"Persisted best model and metadata to {dest_dir}")
            
            return True

        except Exception as e:
            logger.exception(f"Error selecting best model: {e}")
            raise e
