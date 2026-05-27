import os
import joblib
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from kidney_disease_classfication import logger
from kidney_disease_classfication.config.config_entity import ModelTrainerConfig
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        """
        Loads the train and test split datasets, and trains the model chosen
        in params.yaml (LogisticRegression, SVM, MultinomialNB, RandomForest,
        XGBoost, GlobalPoolingFFN, or DistilBERT).
        Saves the resulting model and preprocessors to disk.
        Uses lazy imports and lightweight sample limits for neural networks to keep CPU runs fast.
        """
        try:
            logger.info(f"Loading training dataset for training model: {self.config.model_name}")
            train_df = pd.read_csv(self.config.train_data_path)
            train_df['cleaned_text_final'] = train_df['cleaned_text_final'].fillna("")

            # Prepare directories
            model_path = Path(self.config.trained_model_path)
            os.makedirs(model_path.parent, exist_ok=True)

            model_name = self.config.model_name
            hyperparams = self.config.hyperparams
            random_state = self.config.random_state
            max_features = self.config.max_features

            # Target mapping: map [-1.0, 0.0, 1.0] to [0, 1, 2] for Deep Learning & XGBoost support
            y_train_mapped = (train_df['sentiment'] + 1).astype(int).values

            if model_name in ["LogisticRegression", "SVM", "MultinomialNB", "RidgeClassifier", "XGBoost"]:
                X_train = train_df['cleaned_text_final']
                y_train = train_df['sentiment']

                logger.info(f"Vectorizing features using TF-IDF (max_features={max_features})")
                vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
                X_train_vectorized = vectorizer.fit_transform(X_train)

                if model_name == "LogisticRegression":
                    logger.info("Training high-speed Logistic Regression classifier")
                    C = float(hyperparams.get("C", 1.0))
                    solver = str(hyperparams.get("solver", "lbfgs"))
                    max_iter = int(hyperparams.get("max_iter", 50))
                    classifier = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=random_state, n_jobs=-1)

                elif model_name == "SVM":
                    logger.info("Training high-speed Linear Support Vector Machine (LinearSVC) classifier")
                    C = float(hyperparams.get("C", 1.0))
                    loss = str(hyperparams.get("loss", "squared_hinge"))
                    max_iter = int(hyperparams.get("max_iter", 1000))
                    classifier = LinearSVC(C=C, loss=loss, max_iter=max_iter, random_state=random_state)

                elif model_name == "MultinomialNB":
                    logger.info("Training high-speed pre-tuned Multinomial Naive Bayes classifier")
                    # alpha=0.1 provides optimal vocabulary smoothing for tweets
                    classifier = MultinomialNB(alpha=0.1)

                elif model_name == "RidgeClassifier":
                    logger.info("Training high-speed Ridge Classifier")
                    alpha = float(hyperparams.get("alpha", 1.0))
                    classifier = RidgeClassifier(alpha=alpha, random_state=random_state)

                elif model_name == "XGBoost":
                    logger.info("Training high-speed pre-tuned XGBoost classifier")
                    logger.info("Lazy importing xgboost to preserve resource overhead")
                    from xgboost import XGBClassifier

                    # Optimized parameters for fast and high-performance classification
                    classifier = XGBClassifier(
                        n_estimators=80,
                        max_depth=5,
                        learning_rate=0.15,
                        random_state=random_state,
                        n_jobs=-1,
                        eval_metric='mlogloss'
                    )
                    
                    # Fit with mapped labels
                    classifier.fit(X_train_vectorized, y_train_mapped)
                    
                    # Save classifier model
                    joblib.dump(classifier, model_path)
                    logger.info(f"Trained pre-tuned XGBoost model successfully saved at {model_path}")
                    
                    # Save fitted TF-IDF vectorizer as preprocessor.pkl
                    preprocessor_path = Path("artifacts/data_transformation/preprocessor.pkl")
                    os.makedirs(preprocessor_path.parent, exist_ok=True)
                    joblib.dump(vectorizer, preprocessor_path)
                    logger.info(f"Fitted TF-IDF vectorizer saved at {preprocessor_path}")
                    return
                    
                    # Save classifier model
                    joblib.dump(classifier, model_path)
                    logger.info(f"Trained tuned XGBoost model successfully saved at {model_path}")
                    
                    # Save fitted TF-IDF vectorizer as preprocessor.pkl
                    preprocessor_path = Path("artifacts/data_transformation/preprocessor.pkl")
                    os.makedirs(preprocessor_path.parent, exist_ok=True)
                    joblib.dump(vectorizer, preprocessor_path)
                    logger.info(f"Fitted TF-IDF vectorizer saved at {preprocessor_path}")
                    return

                classifier.fit(X_train_vectorized, y_train)
                
                # Save classifier model
                joblib.dump(classifier, model_path)
                logger.info(f"Trained scikit-learn model successfully saved at {model_path}")

                # Save fitted TF-IDF vectorizer as preprocessor.pkl
                preprocessor_path = Path("artifacts/data_transformation/preprocessor.pkl")
                os.makedirs(preprocessor_path.parent, exist_ok=True)
                joblib.dump(vectorizer, preprocessor_path)
                logger.info(f"Fitted TF-IDF vectorizer saved at {preprocessor_path}")

            elif model_name == "GlobalPoolingFFN":
                # Lazy import TensorFlow
                logger.info("Lazy importing TensorFlow for FFN training")
                import tensorflow as tf

                epochs = int(hyperparams.get("epochs", 1))
                batch_size = int(hyperparams.get("batch_size", 64))
                learning_rate = float(hyperparams.get("learning_rate", 0.001))
                embedding_dim = int(hyperparams.get("embedding_dim", 128))
                max_len = int(hyperparams.get("max_len", 128))

                # Subsample slightly for high-speed Keras CPU execution
                max_train_samples = min(5000, len(train_df))
                logger.info(f"Sampling {max_train_samples} entries for high-speed Keras FFN training")
                sampled_train_df = train_df.sample(n=max_train_samples, random_state=random_state)
                X_train_raw = sampled_train_df['cleaned_text_final'].astype(str).tolist()
                X_train_str = np.array([s.encode('cp1252', errors='ignore').decode('cp1252') for s in X_train_raw], dtype=object)
                y_train_mapped_sampled = (sampled_train_df['sentiment'] + 1).astype(int).values

                logger.info("Initializing Global Pooling Feed-Forward Network in Keras")
                vectorize_layer = tf.keras.layers.TextVectorization(
                    max_tokens=max_features,
                    output_mode='int',
                    output_sequence_length=max_len
                )
                vectorize_layer.adapt(X_train_str)

                model = tf.keras.Sequential([
                    vectorize_layer,
                    tf.keras.layers.Embedding(max_features, embedding_dim, input_length=max_len),
                    tf.keras.layers.GlobalAveragePooling1D(),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(3, activation='softmax')
                ])

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )

                logger.info(f"Training Keras FFN model (epochs={epochs}, batch_size={batch_size})")
                model.fit(
                    X_train_str,
                    y_train_mapped_sampled,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1
                )

                # Save Keras Model
                keras_model_path = str(model_path).replace(".pkl", ".keras")
                model.save(keras_model_path)
                logger.info(f"Keras FFN model successfully saved at {keras_model_path}")

            elif model_name == "DistilBERT":
                # Lazy import TensorFlow and Transformers
                logger.info("Lazy importing TensorFlow and Transformers for DistilBERT training")
                import tensorflow as tf
                from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

                epochs = int(hyperparams.get("epochs", 1))
                batch_size = int(hyperparams.get("batch_size", 32))
                learning_rate = float(hyperparams.get("learning_rate", 2e-5))
                max_len = int(hyperparams.get("max_len", 128))

                # Limit to 100 records for fast and safe fine-tuning on CPU without HUGE resource usage
                sample_size = min(100, len(train_df))
                logger.info(f"Sampling {sample_size} records to train DistilBERT safely under hardware constraints")
                sampled_train_df = train_df.sample(n=sample_size, random_state=random_state)
                
                X_train_str = sampled_train_df['cleaned_text_final'].astype(str).tolist()
                y_train_mapped_sampled = (sampled_train_df['sentiment'] + 1).astype(int).values

                logger.info("Loading DistilBERT tokenizer and TF sequence model")
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

                logger.info("Tokenizing text for DistilBERT")
                train_encodings = tokenizer(X_train_str, truncation=True, padding=True, max_length=max_len, return_tensors="tf")
                
                train_dataset = tf.data.Dataset.from_tensor_slices((
                    dict(train_encodings),
                    y_train_mapped_sampled
                )).shuffle(50).batch(batch_size)

                # Compile model
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                )

                logger.info("Fine-tuning DistilBERT model (epochs=1 on CPU)")
                model.fit(train_dataset, epochs=1)

                # Save Hugging Face model and tokenizer
                model.save_pretrained(str(model_path))
                tokenizer.save_pretrained(str(model_path))
                logger.info(f"DistilBERT model and tokenizer successfully saved at {model_path}")

            else:
                raise ValueError(f"Invalid model name '{model_name}' specified in params.yaml!")

        except Exception as e:
            logger.exception(f"Error encountered during model training stage: {e}")
            raise e
