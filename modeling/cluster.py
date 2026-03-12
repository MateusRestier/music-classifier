"""
cluster.py — Clusterização exploratória do dataset de features musicais.

Uso:
    python modeling/cluster.py [--input PATH] [--k K] [--output-dir DIR]

Pipeline:
    1. Carrega features.parquet
    2. Normaliza com StandardScaler
    3. Reduz dimensionalidade com PCA (95% variância)
    4. Avalia k=2..10 com Elbow + Silhouette Score
    5. Aplica K-Means com k ótimo (ou --k fornecido)
    6. Projeta com t-SNE para visualização 2D
    7. Salva plots em modeling/plots/
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend não-interativo (sem janela)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

DEFAULT_INPUT     = Path(__file__).resolve().parent.parent / "features.parquet"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"

META_COLS = {"file_path", "label", "title", "url"}

# Cores geradas dinamicamente no run_clustering() a partir dos labels presentes no parquet

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


logger = _setup_logging("cluster")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_clustering(input_path: Path, output_dir: Path, k_forced: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar dados
    logger.info("[LOAD] Carregando %s ...", input_path)
    df = pd.read_parquet(input_path)
    logger.info("       Shape: %d linhas × %d colunas", *df.shape)

    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values
    labels = df["label"]

    # 2. Normalizar
    logger.info("[SCALE] StandardScaler ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. PCA
    logger.info("[PCA] Reduzindo dimensionalidade (95%% variância) ...")
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    logger.info("      %d componentes retidos (de %d features)", X_pca.shape[1], X_scaled.shape[1])

    # 4. Elbow + Silhouette (k = 2..10)
    k_range = range(2, 11)
    inertias, silhouettes = [], []

    logger.info("[EVAL] Testando k=2..10 ...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        preds = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_pca, preds, sample_size=min(1000, len(X_pca)), random_state=42)
        silhouettes.append(sil)
        logger.info("  k=%d | inertia=%.0f | silhouette=%.4f", k, km.inertia_, sil)

    # Plot Elbow + Silhouette
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, "o-", color="#3498db")
    axes[0].set_title("Elbow — Inércia por k")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inércia")
    axes[1].plot(list(k_range), silhouettes, "o-", color="#e74c3c")
    axes[1].set_title("Silhouette Score por k")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette Score")
    fig.tight_layout()
    elbow_path = output_dir / "elbow_silhouette.png"
    fig.savefig(elbow_path, dpi=150)
    plt.close(fig)
    logger.info("[SAVE] %s", elbow_path)

    # 5. K-Means com k ótimo
    best_k = k_forced if k_forced is not None else (int(np.argmax(silhouettes)) + 2)
    logger.info("[KMEANS] Ajustando K-Means com k=%d ...", best_k)
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(X_pca)

    # 6. t-SNE
    logger.info("[TSNE] Projetando em 2D (pode demorar alguns minutos) ...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_2d = tsne.fit_transform(X_pca)

    # 7. Scatter plots: clusters vs. labels reais
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Por cluster
    cmap = plt.cm.tab10(np.linspace(0, 1, best_k))
    for ci in range(best_k):
        mask = cluster_labels == ci
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        color=cmap[ci], label=f"Cluster {ci}", s=10, alpha=0.7)
    axes[0].set_title(f"t-SNE — Clusters K-Means (k={best_k})")
    axes[0].legend(markerscale=2, fontsize=8)

    # Por label real — cores geradas automaticamente para todos os gêneros presentes
    unique_labels = sorted(labels.unique())
    label_colors = {
        lbl: plt.cm.tab20(i / max(len(unique_labels) - 1, 1))
        for i, lbl in enumerate(unique_labels)
    }
    for lbl, color in label_colors.items():
        mask = labels == lbl
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        color=color, label=lbl, s=10, alpha=0.7)
    axes[1].set_title("t-SNE — Labels reais")
    axes[1].legend(markerscale=2, fontsize=7, ncol=2)

    fig.tight_layout()
    tsne_path = output_dir / f"tsne_k{best_k}.png"
    fig.savefig(tsne_path, dpi=150)
    plt.close(fig)
    logger.info("[SAVE] %s", tsne_path)

    # Distribuição de labels por cluster
    df_result = df[["title", "label"]].copy()
    df_result["cluster"] = cluster_labels
    logger.info("\n[DIST] Distribuição de labels por cluster:")
    ct = pd.crosstab(df_result["cluster"], df_result["label"])
    logger.info("\n%s\n", ct.to_string())

    logger.info("[DONE] Plots salvos em: %s", output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clusterização exploratória do dataset musical.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=DEFAULT_INPUT,
        help=f"Parquet de features (padrão: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Forçar k clusters (padrão: escolhido pelo Silhouette Score)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Diretório para salvar os plots (padrão: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_clustering(input_path=args.input, output_dir=args.output_dir, k_forced=args.k)
