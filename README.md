# Tweet Sentiment Classification & MLOps Pipeline

This repository implements a production-grade, end-to-end Machine Learning pipeline for **3-Class Tweet Sentiment Classification** (Negative = `-1.0`, Neutral = `0.0`, Positive = `1.0`).

The architecture follows strict MLOps principles, utilizing **MLflow** and **DagsHub** for independent experiment tracking, per-model parameter logging, automated confusion matrix logging, local performance ledgers, and dynamic model comparisons.

---

## 🚀 Key Features

* **Multi-Model Support**: Dynamically train and compare multiple machine learning and deep learning architectures:
  * **Logistic Regression + TF-IDF** (High-speed baseline)
  * **Linear Support Vector Machine (LinearSVC) + TF-IDF** (Max-margin classifier)
  * **Multinomial Naive Bayes + TF-IDF** (Probabilistic classifier)
  * **Random Forest + TF-IDF** (Ensemble baseline)
  * **XGBoost + TF-IDF** (Gradient-boosted decision trees)
  * **Global Pooling FFN** (Keras-based Deep Learning Feed-Forward Network)
  * **DistilBERT** (Hugging Face Contextual Transformer)
* **Isolated Experiment Tracking**: Every model trained is registered inside its own independent, named MLflow run (e.g. `Run_LogisticRegression`), preventing overwrites.
* **Per-Model Parameter & Metrics Logging**: Common and model-specific hyperparameters are logged dynamically from `params.yaml`, alongside macro/weighted metrics.
* **Visual Evaluation Artifacts**: Automatically generates and logs a **Confusion Matrix Heatmap** (`.png`) and a **Classification Report** (`.txt`) for each model straight into MLflow.
* **Consolidated Local Ledger**: Automatically appends and contrasts results locally in `artifacts/model_evaluation/all_models_summary.csv` for fast offline comparisons.
* **Optimized Execution Engine**: Uses lazy imports and smart subsampling to guarantee the pipeline runs quickly and smoothly on standard CPU hardware without OOM risks.
* **Continuous Integration**: Correct, cross-platform GitHub Actions CI workflow to test pipeline compilation and execution on every push.

---

## 📁 Repository Structure

```text
├── .github/workflows/   # GitHub Actions CI workflow (github-actions.yml)
├── config/              # Root-level configuration yaml files
├── research/            # Jupyter Notebooks for exploration
├── src/                 # Main modular python package
│   └── kidney_disease_classfication/
│       ├── components/  # Core modular stages (Ingestion, Transformation, Trainer, Evaluator)
│       ├── config/      # Dataclasses and Configuration Managers
│       ├── constants/   # Path definitions
│       └── utils/       # Common helper utilities
├── .env                 # Environment variables for DagsHub credentials
├── main.py              # Main execution entry-point orchestrator
├── params.yaml          # Hyperparameters and model selection
├── requirements.txt     # Python dependency configuration
└── setup.py             # Package distribution configuration
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/abhishekkamble12/Kidney_diesease-classificiation.git
   cd Kidney_diesease-classificiation
   ```

2. **Set Up Python Virtual Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure MLflow DagsHub Credentials**:
   Open the `.env` file at the root and provide your actual DagsHub password/token:
   ```env
   MLFLOW_TRACKING_URI=https://dagshub.com/abhishekkamble12/Kidney_diesease-classificiation.mlflow
   MLFLOW_TRACKING_USERNAME=abhishekkamble12
   MLFLOW_TRACKING_PASSWORD=YOUR_ACTUAL_DAGSHUB_TOKEN_OR_PASSWORD
   ```

---

## 📈 Running the Pipeline

1. **Configure Hyperparameters**:
   Modify [params.yaml](params.yaml) to select which models to train and customize their parameters:
   ```yaml
   models_to_train:
     - LogisticRegression
     - SVM
     - MultinomialNB
     - RandomForest
     - XGBoost
     - GlobalPoolingFFN
     - DistilBERT
   ```

2. **Execute the End-to-End Pipeline**:
   ```bash
   python main.py
   ```

3. **Analyze Results**:
   * **In the Cloud**: Go to your DagsHub repository's **Experiments** tab to see your separate model runs side-by-side! Compare accuracy, hyperparameters, and view confusion matrix plots.
   * **Locally**: Open the automatically generated CSV summary table at `artifacts/model_evaluation/all_models_summary.csv`.