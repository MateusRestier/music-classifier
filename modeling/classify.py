"""
classify.py — Classificação supervisionada de gêneros musicais.

Uso:
    python modeling/classify.py [--input PATH] [--output-dir DIR]

Pipeline:
    1. Carrega features.parquet
    2. Normaliza com StandardScaler
    3. Split 80/20 estratificado por label
    4. Treina Random Forest e XGBoost
    5. Exibe classification report e confusion matrix por modelo
    6. Salva o melhor modelo em models/classifier.pkl
       junto com models/scaler.pkl e models/label_encoder.pkl

Saídas em modeling/plots/:
    confusion_matrix_random_forest.png
    confusion_matrix_xgboost.png
    feature_importance_rf.png
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

DEFAULT_INPUT      = Path(__file__).resolve().parent.parent / "features.parquet"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
MODELS_DIR         = Path(__file__).resolve().parent.parent / "models"

META_COLS = {"file_path", "label", "title", "url"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(script_name: str) -> logging.Logger:
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{script_name}_{timestamp}.log"

    _logger = logging.getLogger(script_name)
    _logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    _logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(sh)

    return _logger


logger = _setup_logging("classify")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    cm: np.ndarray, classes: list[str], title: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Real")
    ax.set_xlabel("Predito")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("[SAVE] %s", path)


def _evaluate(
    model, X_test: np.ndarray, y_test: np.ndarray,
    label_names: list[str], model_name: str, output_dir: Path,
) -> float:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    logger.info("\n[%s] Accuracy: %.4f | F1-macro: %.4f", model_name, acc, f1)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=label_names))

    cm = confusion_matrix(y_test, y_pred)
    safe_name = model_name.lower().replace(" ", "_")
    _plot_confusion_matrix(
        cm, label_names,
        title=f"Confusion Matrix — {model_name}",
        path=output_dir / f"confusion_matrix_{safe_name}.png",
    )
    return f1


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_classification(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carregar dados
    logger.info("[LOAD] Carregando %s ...", input_path)
    df = pd.read_parquet(input_path)
    logger.info("       Shape: %d linhas × %d colunas", *df.shape)

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values
    y_raw = df["label"].values

    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    label_names = list(le.classes_)

    logger.info("[INFO] Classes: %s", label_names)
    for name in label_names:
        logger.info("       %-15s %d amostras", name, (y_raw == name).sum())

    # 2. Normalizar
    logger.info("\n[SCALE] StandardScaler ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("[SPLIT] Treino: %d | Teste: %d", len(X_train), len(X_test))

    results: dict[str, tuple] = {}  # model_name → (f1, model)

    # 4a. Random Forest
    logger.info("\n[RF] Treinando Random Forest (n_estimators=300) ...")
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    f1_rf = _evaluate(rf, X_test, y_test, label_names, "Random Forest", output_dir)
    results["Random Forest"] = (f1_rf, rf)

    # Feature importance (top 20)
    importances = rf.feature_importances_
    top_n = 20
    top_idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_cols[i] for i in top_idx], importances[top_idx], color="#3498db")
    ax.set_title(f"Top {top_n} Features — Random Forest")
    ax.set_xlabel("Importância")
    fig.tight_layout()
    fi_path = output_dir / "feature_importance_rf.png"
    fig.savefig(fi_path, dpi=150)
    plt.close(fig)
    logger.info("[SAVE] %s", fi_path)

    # 4b. XGBoost
    logger.info("\n[XGB] Treinando XGBoost (n_estimators=300) ...")
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    f1_xgb = _evaluate(xgb, X_test, y_test, label_names, "XGBoost", output_dir)
    results["XGBoost"] = (f1_xgb, xgb)

    # 5. Exportar melhor modelo
    best_name, (best_f1, best_model) = max(results.items(), key=lambda x: x[1][0])
    logger.info("\n[BEST] Melhor modelo: %s (F1-macro=%.4f)", best_name, best_f1)

    scaler_path = MODELS_DIR / "scaler.pkl"
    clf_path    = MODELS_DIR / "classifier.pkl"
    le_path     = MODELS_DIR / "label_encoder.pkl"

    joblib.dump(scaler, scaler_path)
    joblib.dump(best_model, clf_path)
    joblib.dump(le, le_path)
    logger.info("[SAVE] %s", scaler_path)
    logger.info("[SAVE] %s", clf_path)
    logger.info("[SAVE] %s", le_path)

    logger.info("\n[DONE] Resultados finais:")
    for name, (f1, _) in results.items():
        logger.info("  %-15s F1-macro=%.4f", name, f1)
    logger.info("\nModelo exportado: %s", clf_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classificação supervisionada de gêneros musicais.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=DEFAULT_INPUT,
        help=f"Parquet de features (padrão: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Diretório para salvar os plots (padrão: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_classification(input_path=args.input, output_dir=args.output_dir)
