import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from sklearn.datasets import load_iris, load_breast_cancer, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib


def _maybe_mlflow():
    """Best-effort import of mlflow; returns None if unavailable."""
    try:
        import mlflow  # type: ignore

        return mlflow
    except Exception:
        return None


@dataclass
class TrainResult:
    """Container for training results."""
    model_path: str
    metrics: Dict[str, float]
    params: Dict[str, Any]
    run_id: Optional[str] = None


def load_data(
    data_path: Optional[str] = None,
    target: Optional[str] = None,
    task: str = "classification",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset from CSV or fall back to a built-in toy dataset.

    Parameters
    ----------
    data_path : str, optional
        Path to a CSV file. If None or not found, a toy dataset is used.
    target : str, optional
        Name of the target column for CSV input. If None, 'target' is assumed.
    task : {'classification', 'regression'}
        Type of ML task.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target vector.
    """
    if data_path and pd is not None and Path(data_path).exists():
        df = pd.read_csv(data_path)
        tgt = target or "target"
        if tgt not in df.columns:
            raise ValueError(f"Target column '{tgt}' not found in {data_path}.")
        y = df[tgt].values
        X = df.drop(columns=[tgt]).values
        return X, y

    # Toy datasets as fallback to keep the starter repo runnable
    if task == "classification":
        data = load_iris() if random.random() < 0.5 else load_breast_cancer()
        return data.data, data.target
    else:
        X, y = make_regression(n_samples=800, n_features=12, noise=12.0, random_state=0)
        return X, y


def build_model(task: str = "classification", model: str = "logreg"):
    """Build a scikit-learn model for the given task.

    Parameters
    ----------
    task : {'classification', 'regression'}
        Type of ML task.
    model : {'logreg', 'rf', 'linreg'}
        Model choice. For regression, 'linreg' and 'rf' are supported. For classification, 'logreg' and 'rf'.

    Returns
    -------
    estimator : sklearn.base.BaseEstimator
        Instantiated scikit-learn model.
    """
    if task == "classification":
        if model == "rf":
            return RandomForestClassifier(n_estimators=200, random_state=0)
        return LogisticRegression(max_iter=1000, n_jobs=None)
    else:
        if model == "rf":
            return RandomForestRegressor(n_estimators=300, random_state=0)
        return LinearRegression()


def evaluate(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute basic metrics for the given task.

    Parameters
    ----------
    task : {'classification', 'regression'}
        Type of ML task.
    y_true : ndarray
        Ground-truth labels/values.
    y_pred : ndarray
        Predicted labels/values.

    Returns
    -------
    metrics : dict
        Dictionary of metric name -> value.
    """
    if task == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        }
    else:
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }


def train(
    data_path: Optional[str] = None,
    target: Optional[str] = None,
    task: str = "classification",
    model: str = "logreg",
    test_size: float = 0.2,
    random_state: int = 42,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "mlops-starter",
    output_dir: str = "artifacts",
) -> TrainResult:
    """Train a simple ML model, evaluate it, and optionally log to MLflow.

    Parameters
    ----------
    data_path : str, optional
        Path to CSV dataset. If None or missing, a toy dataset is used.
    target : str, optional
        Target column name for CSV datasets. Defaults to 'target' if None.
    task : {'classification', 'regression'}, default='classification'
        Type of ML task.
    model : {'logreg', 'rf', 'linreg'}, default='logreg'
        Model choice.
    test_size : float, default=0.2
        Test split fraction.
    random_state : int, default=42
        Random seed for splitting and models.
    tracking_uri : str, optional
        MLflow tracking URI. If None, uses MLflow default or skips logging if MLflow unavailable.
    experiment_name : str, default='mlops-starter'
        MLflow experiment name.
    output_dir : str, default='artifacts'
        Directory to store the serialized model.

    Returns
    -------
    result : TrainResult
        Paths, metrics, params, and MLflow run_id (if any).
    """
    np.random.seed(random_state)
    random.seed(random_state)

    X, y = load_data(data_path=data_path, target=target, task=task)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if task == "classification" else None
    )

    estimator = build_model(task=task, model=model)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    metrics = evaluate(task, y_test, y_pred)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"model_{task}_{model}.joblib")
    joblib.dump(estimator, model_path)

    params = {
        "task": task,
        "model": model,
        "test_size": test_size,
        "random_state": random_state,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
        "data_path": str(data_path) if data_path else "builtin",
        "target": target or "target",
    }

    run_id = None
    mlflow = _maybe_mlflow()
    if mlflow is not None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"{model}-{task}") as run:
            run_id = run.info.run_id
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            try:
                import mlflow.sklearn as msk  # type: ignore

                msk.log_model(estimator, artifact_path="model")
            except Exception:
                mlflow.log_artifact(model_path, artifact_path="model_file")

    return TrainResult(model_path=model_path, metrics=metrics, params=params, run_id=run_id)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train a simple ML model and optionally log to MLflow.")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset. If omitted, uses a toy dataset.")
    parser.add_argument("--target", type=str, default=None, help="Target column name for CSV datasets.")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], default="classification")
    parser.add_argument("--model", type=str, choices=["logreg", "rf", "linreg"], default="logreg")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI.")
    parser.add_argument("--experiment", type=str, default="mlops-starter", help="MLflow experiment name.")
    parser.add_argument("--out", type=str, default="artifacts", help="Output dir for serialized model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train(
        data_path=args.data,
        target=args.target,
        task=args.task,
        model=args.model,
        test_size=args.test_size,
        random_state=args.seed,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
        output_dir=args.out,
    )
    print("Model saved to:", result.model_path)
    print("Metrics:", result.metrics)
    if result.run_id:
        print("MLflow run_id:", result.run_id)
