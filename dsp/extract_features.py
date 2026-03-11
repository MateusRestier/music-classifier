"""
extract_features.py — Extração de features de áudio via librosa.

Uso:
    python dsp/extract_features.py [--workers N] [--output PATH]

Lê os caminhos dos arquivos .wav do MongoDB, extrai features numéricas
e salva um DataFrame em features.parquet (uma linha por faixa).

Features extraídas (por segmento de 30s, resumidas com mean + std):
    MFCCs (40 coef.)           → mean + std  (80 colunas)
    Mel-Spectrogram (128 bins) → mean + std  (256 colunas)
    Chroma (12 bins)           → mean + std  (24 colunas)
    Spectral Centroid          → mean + std  (2 colunas)
    Spectral Rolloff           → mean + std  (2 colunas)
    Zero-Crossing Rate         → mean + std  (2 colunas)
    RMS Energy                 → mean + std  (2 colunas)
    Tempo (BPM)                →             (1 coluna)
                                             ──────────
                               Total:        369 features
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from pymongo import MongoClient
from pymongo.collection import Collection

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:admin123@localhost:27017/")
DB_NAME = "music_classifier"
COLLECTION_NAME = "tracks"

SR = 22050               # sample rate alvo
SEGMENT_OFFSET = 30.0    # segundos a pular no início (evitar intros)
SEGMENT_DURATION = 30.0  # segundos do segmento a analisar
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512

DEFAULT_OUTPUT = Path("features.parquet")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))


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

    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(file_fmt)
    _logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(sh)

    return _logger


logger = _setup_logging("extract_features")


# ---------------------------------------------------------------------------
# Extração de features (reutilizável no Streamlit)
# ---------------------------------------------------------------------------

def extract_features_from_wav(file_path: str | Path) -> dict:
    """
    Carrega um segmento do arquivo .wav e extrai features de áudio.

    Tenta carregar SEGMENT_DURATION segundos a partir de SEGMENT_OFFSET.
    Se o arquivo for mais curto que o offset, carrega do início.

    Retorna dict com todas as features numéricas (369 valores).
    Lança ValueError se o arquivo tiver menos de 1 segundo de áudio.
    """
    file_path = str(file_path)

    # Tenta carregar a partir do offset; fallback para início se o arquivo for curto
    y, sr = librosa.load(file_path, sr=SR, offset=SEGMENT_OFFSET,
                         duration=SEGMENT_DURATION, mono=True)
    if len(y) < SR:
        y, sr = librosa.load(file_path, sr=SR, duration=SEGMENT_DURATION, mono=True)
    if len(y) < SR:
        raise ValueError(f"Áudio muito curto: {len(y) / sr:.1f}s")

    features: dict = {}

    # --- MFCCs ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    for i in range(N_MFCC):
        features[f"mfcc_mean_{i+1:02d}"] = float(np.mean(mfcc[i]))
        features[f"mfcc_std_{i+1:02d}"]  = float(np.std(mfcc[i]))

    # --- Mel-Spectrogram (em dB) ---
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    for i in range(N_MELS):
        features[f"mel_mean_{i+1:03d}"] = float(np.mean(mel_db[i]))
        features[f"mel_std_{i+1:03d}"]  = float(np.std(mel_db[i]))

    # --- Chroma ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    for i in range(12):
        features[f"chroma_mean_{i+1:02d}"] = float(np.mean(chroma[i]))
        features[f"chroma_std_{i+1:02d}"]  = float(np.std(chroma[i]))

    # --- Spectral Centroid ---
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"]  = float(np.std(centroid))

    # --- Spectral Rolloff ---
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"]  = float(np.std(rolloff))

    # --- Zero-Crossing Rate ---
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"]  = float(np.std(zcr))

    # --- RMS Energy ---
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"]  = float(np.std(rms))

    # --- Tempo (BPM) ---
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    features["tempo"] = float(np.atleast_1d(tempo)[0])

    return features


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def get_tracks(collection: Collection) -> list[dict]:
    """Retorna todas as faixas do MongoDB que têm file_path registrado."""
    return list(collection.find(
        {"file_path": {"$ne": ""}},
        {"_id": 0, "title": 1, "url": 1, "label": 1, "file_path": 1},
    ))


def _process_track(track: dict) -> dict | None:
    """Extrai features de uma faixa. Retorna None em caso de erro."""
    file_path = Path(track["file_path"])
    if not file_path.exists():
        logger.warning("  [SKIP] Arquivo não encontrado: %s", file_path)
        return None

    try:
        features = extract_features_from_wav(file_path)
    except Exception as exc:
        logger.error("  [ERR]  %s — %s", file_path.name, exc)
        return None

    return {
        "file_path": str(file_path),
        "label":     track["label"],
        "title":     track["title"],
        "url":       track["url"],
        **features,
    }


def run_extraction(output_path: Path, max_workers: int) -> None:
    """Pipeline principal: lê MongoDB → extrai features → salva parquet."""
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    all_tracks = get_tracks(collection)
    logger.info("[INFO] %d faixa(s) encontrada(s) no banco.", len(all_tracks))

    # Incremental: pula faixas já presentes no parquet existente
    if output_path.exists():
        existing = pd.read_parquet(output_path, columns=["file_path"])
        done_paths = set(existing["file_path"].tolist())
        pending = [t for t in all_tracks if t["file_path"] not in done_paths]
        logger.info(
            "[INFO] %d já processada(s), %d na fila.",
            len(done_paths), len(pending),
        )
    else:
        pending = all_tracks
        done_paths = set()

    if not pending:
        logger.info("Nenhuma faixa nova para processar.")
        return

    logger.info(
        "\n[START] Extraindo features de %d faixa(s) com %d workers...\n",
        len(pending), max_workers,
    )

    rows = []
    done = errors = 0
    total = len(pending)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_track, t): t for t in pending}
        for future in as_completed(futures):
            track = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error("  [ERR]  Exceção inesperada em %r: %s", track["title"], exc)
                errors += 1
                result = None

            if result is not None:
                rows.append(result)
                done += 1
            else:
                errors += 1

            logger.info(
                "  [PROG] %d/%d concluídos (%d erro(s)) — %s",
                done + errors, total, errors, track["title"],
            )

    if not rows:
        logger.warning("Nenhuma feature extraída. Verifique os arquivos .wav.")
        return

    new_df = pd.DataFrame(rows)

    # Concatena com o parquet existente se houver
    if output_path.exists() and done_paths:
        existing_df = pd.read_parquet(output_path)
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df.to_parquet(output_path, index=False)
    logger.info(
        "\nConcluído. %d extraída(s), %d erro(s). Parquet salvo em: %s",
        done, errors, output_path,
    )
    logger.info("Shape final: %d linhas × %d colunas.", *final_df.shape)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extração de features de áudio para o music-classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=MAX_WORKERS,
        help=f"Número de workers paralelos (padrão: {MAX_WORKERS})",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=DEFAULT_OUTPUT,
        help=f"Caminho do arquivo de saída (padrão: {DEFAULT_OUTPUT})",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_extraction(output_path=args.output, max_workers=args.workers)
