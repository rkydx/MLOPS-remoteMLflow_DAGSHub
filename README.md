# MLOPS-remoteMLflow_DAGSHub

This repository demonstrates how to track machine learning experiments remotely using **MLflow** integrated with **DAGsHub**. It trains a simple `RandomForestClassifier` on the Wine dataset and logs key artifacts such as parameters, metrics, model files, and visualizations to the DAGsHub MLflow tracking server.

## Features

- Uses **scikit-learn** to train a Random Forest model on the Wine dataset.
- Tracks experiments using **MLflow**.
- Stores all experiment logs, metrics, artifacts, and models in **DAGsHub**.
- Logs include:
  - Model parameters (e.g., `max_depth`, `n_estimators`)
  - Accuracy score
  - Confusion matrix (as image)
  - Full training script

## How to Run

1. **Install Dependencies**
   ```bash
   pip install "mlflow<3" seaborn dagshub scikit-learn matplotlib

2. **Run the Experiment**
    ```bash
    python experiment_remoteDAGSHub.py

3. **Check your DAGSHub Dashboard**
    After the run completes, visit your [DAGSHub MLFlow UI](https://dagshub.com/rkydx/MLOPS-remoteMLflow_DAGSHub.mlflow) to explore the run details.

## Files
 - experiment_remoteDAGSHub.py: Main script for training and logging the model.
 - confusion_matrix.png: Logged as an artifact to visualize prediction performance.
 - README.md: Explains the repository's purpose and usage.