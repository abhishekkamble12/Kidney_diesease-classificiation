import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from kidney_disease_classfication import logger
from kidney_disease_classfication.config.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        """
        Loads the train and test split datasets, and trains the model chosen
        in params.yaml (LogisticRegression, MultinomialNB, GlobalPoolingFFN, or DistilBERT).
        Saves the resulting model and preprocessors to disk.
        Uses lazy imports and lightweight sample limits to save CPU resources and run quickly.
        """
        try:
            logger.info("Loading training and testing datasets for model training")
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            # Fill missing text values to prevent Vectorizer crashes
            logger.info("Cleaning missing values in text features")
            train_df['cleaned_text_final'] = train_df['cleaned_text_final'].fillna("")
            test_df['cleaned_text_final'] = test_df['cleaned_text_final'].fillna("")

            # Prepare directories
            model_path = Path(self.config.trained_model_path)
            os.makedirs(model_path.parent, exist_ok=True)

            model_name = self.config.model_name
            logger.info(f"Selected model for training: {model_name}")

            # Map labels (-1.0 -> 0, 0.0 -> 1, 1.0 -> 2) for deep learning classification
            y_train_mapped = (train_df['sentiment'] + 1).astype(int).values

            if model_name in ["LogisticRegression", "MultinomialNB"]:
                X_train = train_df['cleaned_text_final']
                y_train = train_df['sentiment']

                logger.info(f"Vectorizing features using TF-IDF (max_features={self.config.max_features})")
                vectorizer = TfidfVectorizer(max_features=self.config.max_features, ngram_range=(1, 2))
                X_train_vectorized = vectorizer.fit_transform(X_train)

                if model_name == "LogisticRegression":
                    logger.info("Training high-speed Logistic Regression classifier")
                    classifier = LogisticRegression(max_iter=50, C=1.0, random_state=42, n_jobs=-1, solver='lbfgs')
                else:
                    logger.info("Training high-speed Multinomial Naive Bayes classifier")
                    classifier = MultinomialNB(alpha=1.0)

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
                # Lazy import tensorflow to keep Logistic Regression / Naive Bayes starts instant
                logger.info("Lazy importing TensorFlow to preserve computational resources")
                import tensorflow as tf

                # Subsample dataset slightly for super-fast Keras training on CPU
                max_train_samples = min(5000, len(train_df))
                logger.info(f"Sampling {max_train_samples} entries for high-speed Keras CPU training")
                sampled_train_df = train_df.sample(n=max_train_samples, random_state=42)
                X_train_str = sampled_train_df['cleaned_text_final'].astype(str).tolist()
                y_train_mapped_sampled = (sampled_train_df['sentiment'] + 1).astype(int).values

                logger.info("Initializing Global Pooling Feed-Forward Network in Keras")
                vectorize_layer = tf.keras.layers.TextVectorization(
                    max_tokens=self.config.max_features,
                    output_mode='int',
                    output_sequence_length=self.config.max_len
                )
                vectorize_layer.adapt(X_train_str)

                model = tf.keras.Sequential([
                    vectorize_layer,
                    tf.keras.layers.Embedding(self.config.max_features, self.config.embedding_dim, input_length=self.config.max_len),
                    tf.keras.layers.GlobalAveragePooling1D(),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(3, activation='softmax')
                ])

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )

                # Keep epochs very low (e.g. 1 epoch) to train in 2 seconds!
                logger.info(f"Training Keras FFN model (epochs=1, batch_size={self.config.batch_size})")
                model.fit(
                    X_train_str,
                    y_train_mapped_sampled,
                    epochs=1,
                    batch_size=self.config.batch_size,
                    verbose=1
                )

                # Save Keras Model
                model.save(str(model_path))
                logger.info(f"Keras FFN model successfully saved at {model_path}")

            elif model_name == "DistilBERT":
                # Lazy import TensorFlow and Transformers
                logger.info("Lazy importing TensorFlow and Transformers to preserve CPU resources")
                import tensorflow as tf
                from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

                # Limit to 100 records so it trains in 10-15 seconds on a simple CPU without huge resource usage!
                sample_size = min(100, len(train_df))
                logger.info(f"Sampling {sample_size} records to train DistilBERT safely under hardware constraints")
                sampled_train_df = train_df.sample(n=sample_size, random_state=42)
                
                X_train_str = sampled_train_df['cleaned_text_final'].astype(str).tolist()
                y_train_mapped_sampled = (sampled_train_df['sentiment'] + 1).astype(int).values

                logger.info("Loading DistilBERT tokenizer and TF sequence model")
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

                logger.info("Tokenizing text for DistilBERT")
                train_encodings = tokenizer(X_train_str, truncation=True, padding=True, max_length=self.config.max_len, return_tensors="tf")
                
                # Convert encodings to tf dataset
                train_dataset = tf.data.Dataset.from_tensor_slices((
                    dict(train_encodings),
                    y_train_mapped_sampled
                )).shuffle(50).batch(self.config.batch_size)

                # Compile model
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                )

                logger.info(f"Fine-tuning DistilBERT (epochs=1 for ultra-lightweight training on CPU)")
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
