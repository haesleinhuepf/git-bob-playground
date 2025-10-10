from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Optional Prefect integration with graceful fallback
try:
    from prefect import flow, task, get_run_logger
except Exception:  # pragma: no cover
    def flow(func=None, **kwargs):
        def wrapper(f):
            return f

        return wrapper if func is None else func

    def task(func=None, **kwargs):
        def wrapper(f):
            return f

        return wrapper if func is None else func

    def get_run_logger():
        return logging.getLogger("mlops.flow")


# Optional MLflow integration with graceful fallback
try:
    import mlflow
    import mlflow.sklearn as mlflow_sklearn
except Exception:  # pragma: no cover
    mlflow = None
    mlflow_sklearn = None

# scikit-learn is required for an actual run
try:
    from sklearn import datasets, metrics, model_selection
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    SKLEARN_AVAILABLE = False


@dataclass
class TrainConfig:
    """
    Configuration for the end-to-end ML pipeline.

    Parameters
    ----------
    test_size : float
        Fraction of the dataset used for testing.
    random_state : int
        Random seed for reproducibility.
    model_type : {"logreg", "rf"}
        Type of model to train. "logreg" for Logistic Regression, "rf" for Random Forest.
    c : float
        Inverse regularization strength for Logistic Regression (only if model_type="logreg").
    max_iter : int
        Maximum iterations for Logistic Regression solver.
    n_estimators : int
        Number of trees for Random Forest (only if model_type="rf").
    max_depth : int or None
        Maximum depth of trees for Random Forest (only if model_type="rf").
    experiment_name : str
        MLflow experiment name (used if MLflow is available).
    tracking_uri : str or None
        MLflow tracking URI. If None, defaults to a local ./mlruns folder.
    artifacts_dir : str
        Local directory to write model artifacts (pickle) regardless of MLflow availability.
    """

    test_size: float = 0.2
    random_state: int = 42
    model_type: str = "logreg"  # or "rf"
    c: float = 1.0
    max_iter: int = 200
    n_estimators: int = 200
    max_depth: Optional[int] = None
    experiment_name: str = "mlops-starter"
    tracking_uri: Optional[str] = "file:./mlruns"
    artifacts_dir: str = "artifacts/models"


def _ensure_requirements():
    """
    Verify that required libraries are available; raise actionable errors otherwise.

    Raises
    ------
    ImportError
        If scikit-learn is not available.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for this pipeline.\n"
            "Install with: pip install scikit-learn"
        )


@task
def load_data(cfg: TrainConfig) -> Tuple[Any, Any, Any, Any]:
    """
    Load and split a tabular classification dataset.

    Uses sklearn's breast_cancer dataset for a fast, local example.

    Parameters
    ----------
    cfg : TrainConfig
        Pipeline configuration.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Train/test split of features and labels.
    """
    _ensure_requirements()
    data = datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        data.data,
        data.target,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=data.target,
    )
    return X_train, X_test, y_train, y_test


@task
def train_model(
    cfg: TrainConfig, X_train: Any, y_train: Any
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a model according to configuration.

    Parameters
    ----------
    cfg : TrainConfig
        Pipeline configuration.
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.

    Returns
    -------
    model : Any
        Trained estimator.
    params : dict
        Parameters used for training (logged to MLflow if available).
    """
    _ensure_requirements()
    if cfg.model_type == "logreg":
        model = LogisticRegression(
            C=cfg.c, max_iter=cfg.max_iter, random_state=cfg.random_state, n_jobs=None
        )
        params = {"model_type": "logreg", "C": cfg.c, "max_iter": cfg.max_iter}
    elif cfg.model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        params = {
            "model_type": "rf",
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
        }
    else:
        raise ValueError(f"Unknown model_type={cfg.model_type}")

    model.fit(X_train, y_train)
    return model, params


@task
def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.

    Parameters
    ----------
    model : Any
        Trained estimator.
    X_test : array-like
        Test features.
    y_test : array-like
        Test labels.

    Returns
    -------
    metrics_dict : dict
        Dictionary with accuracy, precision, recall, and f1 scores.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(metrics.accuracy_score(y_test, y_pred)),
        "precision": float(metrics.precision_score(y_test, y_pred)),
        "recall": float(metrics.recall_score(y_test, y_pred)),
        "f1": float(metrics.f1_score(y_test, y_pred)),
    }


@task
def persist_locally(model: Any, cfg: TrainConfig) -> str:
    """
    Persist the trained model as a pickle file in artifacts_dir.

    Parameters
    ----------
    model : Any
        Trained estimator.
    cfg : TrainConfig
        Pipeline configuration.

    Returns
    -------
    path : str
        Path to the saved pickle file.
    """
    Path(cfg.artifacts_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = Path(cfg.artifacts_dir) / f"{cfg.model_type}_{timestamp}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return str(path)


def _setup_mlflow(cfg: TrainConfig):
    if mlflow is None:  # pragma: no cover
        return None
    if cfg.tracking_uri:
        mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)
    return mlflow


def _mlflow_log_run(
    cfg: TrainConfig, params: Dict[str, Any], metrics_dict: Dict[str, float], model: Any
) -> Optional[str]:
    """
    Log params, metrics and model to MLflow if available.

    Returns run_id or None.
    """
    if mlflow is None or mlflow_sklearn is None:  # pragma: no cover
        return None
    _setup_mlflow(cfg)
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_dict(asdict(cfg), "config.json")
        mlflow.log_metrics(metrics_dict)
        mlflow_sklearn.log_model(model, artifact_path="model")
        return run.info.run_id


@flow(name="mlops-starter-flow")
def run_pipeline(cfg: TrainConfig = TrainConfig()) -> Dict[str, Any]:
    """
    End-to-end example ML pipeline: load -> train -> evaluate -> log -> persist.

    Parameters
    ----------
    cfg : TrainConfig, optional
        Pipeline configuration.

    Returns
    -------
    result : dict
        Contains metrics, artifact path, and optionally MLflow run_id.
    """
    logger = get_run_logger()
    logger.info("Starting pipeline with config: %s", json.dumps(asdict(cfg), indent=2))

    X_train, X_test, y_train, y_test = load_data(cfg)
    model, params = train_model(cfg, X_train, y_train)
    metrics_dict = evaluate_model(model, X_test, y_test)

    run_id = _mlflow_log_run(cfg, params, metrics_dict, model)
    artifact_path = persist_locally(model, cfg)

    logger.info("Metrics: %s", metrics_dict)
    logger.info("Model saved to: %s", artifact_path)
    if run_id:
        logger.info("MLflow run_id: %s", run_id)

    return {
        "metrics": metrics_dict,
        "artifact_path": artifact_path,
        "mlflow_run_id": run_id,
    }


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Run the MLOps starter pipeline.")
    parser.add_argument("--model-type", choices=["logreg", "rf"], default="logreg")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--experiment-name", type=str, default="mlops-starter")
    parser.add_argument("--tracking-uri", type=str, default="file:./mlruns")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/models")

    args = parser.parse_args()
    return TrainConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        model_type=args.model_type,
        c=args.c,
        max_iter=args.max_iter,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        artifacts_dir=args.artifacts_dir,
    )


if __name__ == "__main__":
    # Basic console logging if not run via Prefect UI/agent
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    cfg = _parse_args()
    out = run_pipeline(cfg)
    print(json.dumps(out, indent=2))
