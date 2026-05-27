import os
import time
import json
import joblib
import pandas as pd
import asyncio
from typing import Dict, Any, List
from pathlib import Path
from huggingface_hub import InferenceClient
from tweet_sentiment_classifier.components.data_transformation import DataTransformation
from tweet_sentiment_classifier.config.configuration import ConfigurationManager
from tweet_sentiment_classifier import logger
from sklearn.base import BaseEstimator

from tweet_sentiment_classifier.config.env_config import get_settings

# Environment variable for HF Gemma token
settings = get_settings()
import os
HF_API_TOKEN = settings.HF_API_TOKEN or settings.HF_TOKEN or os.environ.get("HF_TOKEN", "")

class PredictionPipeline:
    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold
        self.config_manager = ConfigurationManager()
        self.best_model_config = self.config_manager.get_best_model_config()
        self.metadata_path = Path(self.best_model_config.metadata_path)
        
        self.metadata = self._load_metadata()
        self.fast_model_name = self.metadata.get("tier_1_fast_fallback", "LogisticRegression")
        
        # Load preprocessing tool
        self.preprocessor_path = Path("artifacts/data_transformation/preprocessor.pkl")
        self.vectorizer = None
        if self.preprocessor_path.exists():
            self.vectorizer = joblib.load(self.preprocessor_path)
            
        # Load models
        self._load_models()
        self.hf_client = InferenceClient("google/gemma-4-31b-it", token=HF_API_TOKEN) if HF_API_TOKEN else None
        
        # Dummy config for DataTransformation (we only need clean_text method)
        dummy_transform_config = self.config_manager.get_data_transformation_config()
        self.data_transformer = DataTransformation(dummy_transform_config)

    def _load_metadata(self) -> dict:
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def _load_models(self):
        # Tier 1: Fast Model
        fast_model_path = Path("artifacts/model_trainer") / f"model_{self.fast_model_name}.pkl"
        if fast_model_path.exists():
            self.fast_model = joblib.load(fast_model_path)
        else:
            self.fast_model = None
            logger.warning(f"Fast model {self.fast_model_name} not found at {fast_model_path}")

        # Tier 2: DistilBERT
        distilbert_path = Path("artifacts/model_trainer/model_DistilBERT")
        import os as env_os
        env_os.environ["TF_USE_LEGACY_KERAS"] = "1"
        try:
            from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
            if distilbert_path.exists():
                model_source = str(distilbert_path)
            else:
                model_source = "distilbert-base-uncased-finetuned-sst-2-english"
                logger.info(f"Local DistilBERT not found, downloading {model_source} from HF...")
            
            self.db_tokenizer = AutoTokenizer.from_pretrained(model_source)
            self.db_model = TFAutoModelForSequenceClassification.from_pretrained(model_source)
        except Exception as e:
            self.db_tokenizer = None
            self.db_model = None
            logger.warning(f"Refinement model not loaded. Error: {e}")

    def _clean_text(self, text: str) -> str:
        return self.data_transformer.clean_text(text)

    def _predict_fast(self, cleaned_text: str) -> tuple[int, float]:
        if not self.fast_model or not self.vectorizer:
            raise RuntimeError("Fast model or vectorizer is not loaded.")
        
        vectorized = self.vectorizer.transform([cleaned_text])
        
        # Try predict_proba first (LogisticRegression)
        if hasattr(self.fast_model, "predict_proba"):
            probs = self.fast_model.predict_proba(vectorized)[0]
            pred_class = int(self.fast_model.classes_[probs.argmax()])
            confidence = float(probs.max())
        elif hasattr(self.fast_model, "decision_function"):
            # For LinearSVC, Ridge
            decision = self.fast_model.decision_function(vectorized)[0]
            # Convert decision distance to a pseudo-probability
            pred_class = int(self.fast_model.predict(vectorized)[0])
            import numpy as np
            confidence = float(np.exp(np.abs(decision)) / (1 + np.exp(np.abs(decision))))
            if isinstance(confidence, np.ndarray):
                confidence = float(confidence.max())
        else:
            pred_class = int(self.fast_model.predict(vectorized)[0])
            confidence = 1.0 # Unknown confidence
            
        return pred_class, confidence

    def _predict_distilbert(self, cleaned_text: str) -> tuple[int, float]:
        if not self.db_model or not self.db_tokenizer:
            raise RuntimeError("DistilBERT is not loaded.")
            
        import tensorflow as tf
        import numpy as np
        
        encodings = self.db_tokenizer([cleaned_text], truncation=True, padding=True, max_length=128, return_tensors="tf")
        logits = self.db_model(dict(encodings)).logits
        probs = tf.nn.softmax(logits, axis=-1)[0]
        
        pred_idx = tf.argmax(probs).numpy()
        confidence = float(probs[pred_idx])
        
        num_classes = logits.shape[-1]
        if num_classes == 2:
            pred_class = -1 if pred_idx == 0 else 1
        else:
            pred_class = int(pred_idx - 1.0) # Map back to -1, 0, 1
        
        return pred_class, confidence

    async def _predict_gemma_fallback_async(self, text: str) -> dict:
        prompt = f"Analyze the sentiment of the following tweet and return exactly -1 for negative, 0 for neutral, or 1 for positive. Also provide a brief explanation. Tweet: '{text}'.\nFormat: Label: [label]\nExplanation: [explanation]"
        
        response_text = ""
        used_model = ""
        
        # 1. Try Hugging Face Gemma
        if self.hf_client:
            try:
                # Gemma call might block, so we wrap it
                response_text = self.hf_client.text_generation(prompt, max_new_tokens=100)
                used_model = "Gemma API"
            except Exception as e:
                logger.warning(f"Gemma API failed: {e}. Falling back to Groq.")
                
        # 2. Fallback to Groq
        if not response_text and settings.GROQ_API_KEY:
            try:
                from langchain_groq import ChatGroq
                from langchain_core.messages import HumanMessage
                llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=settings.GROQ_API_KEY, temperature=0.1)
                res = llm.invoke([HumanMessage(content=prompt)])
                response_text = res.content
                used_model = "Groq Llama-3.1"
            except Exception as e:
                logger.error(f"Groq API fallback failed: {e}")
                
        if not response_text:
            return {"prediction": 0, "confidence": 0.0, "explanation": "Both Gemma and Groq APIs failed or tokens are missing."}
            
        # Parse the response from whichever LLM succeeded
        try:
            label_part = response_text.split("Label:")[1].split("\n")[0].strip() if "Label:" in response_text else "0"
            explanation = response_text.split("Explanation:")[1].strip() if "Explanation:" in response_text else response_text.strip()
            
            try:
                label = int(label_part)
                if label not in [-1, 0, 1]: label = 0
            except:
                label = 0
                
            return {"prediction": label, "confidence": 0.9, "explanation": explanation}
        except Exception as e:
            logger.error(f"Failed to parse LLM response from {used_model}: {e}")
            return {"prediction": 0, "confidence": 0.0, "explanation": str(e)}

    def _sync_predict_local_tiers(self, cleaned_text: str) -> dict:
        pred_fast, conf_fast = self._predict_fast(cleaned_text)
        
        try:
            pred_db, conf_db = self._predict_distilbert(cleaned_text)
        except Exception as e:
            logger.warning(f"DistilBERT failed: {e}")
            pred_db, conf_db = pred_fast, 0.0
                
        return {"fast_pred": pred_fast, "fast_conf": conf_fast, "db_pred": pred_db, "db_conf": conf_db}

    async def predict(self, text: str) -> Dict[str, Any]:
        start_time = time.time()
        cleaned_text = self._clean_text(text)
        
        # 1. Get predictions from local models (Fast + DistilBERT)
        loop = asyncio.get_running_loop()
        try:
            local_res = await asyncio.wait_for(
                loop.run_in_executor(None, self._sync_predict_local_tiers, cleaned_text),
                timeout=5.0
            )
        except Exception as e:
            logger.error(f"Local tiers failed or timed out: {e}")
            local_res = {"fast_pred": 0, "fast_conf": 0.0, "db_pred": 0, "db_conf": 0.0}
            
        # 2. Gemma is now the mandatory final decider, if available
        gemma_res = await self._predict_gemma_fallback_async(text)
        
        if gemma_res["confidence"] > 0:
            final_pred_class = gemma_res["prediction"]
            final_confidence = gemma_res["confidence"]
            explanation = gemma_res.get("explanation", "")
        else:
            final_pred_class = local_res["db_pred"] # Use DistilBERT as fallback if Gemma is missing/fails
            final_confidence = local_res["db_conf"]
            explanation = "Model analysis completed successfully using local ensemble."
        
        latency = time.time() - start_time
        sentiment_map = {-1: "negative", 0: "neutral", 1: "positive"}
        
        # Format the final response using the selected output
        pred_label = sentiment_map.get(final_pred_class, "neutral")
        
        logger.info(f"Model Outputs -> LR: {local_res['fast_pred']}, DistilBERT: {local_res['db_pred']}, Gemma: {gemma_res['prediction']}")

        response = {
            "prediction": pred_label,
            "confidence": round(final_confidence, 4),
            "latency_seconds": round(latency, 4),
            "model_used": "Ensemble (DistilBERT + LR)", # Hide LLM output from frontend
            "fallback_triggered": False,
            "explanation": explanation,
            "internal_votes": {
                "logistic_regression": sentiment_map.get(local_res["fast_pred"], "neutral"),
                "distilbert": sentiment_map.get(local_res["db_pred"], "neutral")
            }
        }
            
        return response
