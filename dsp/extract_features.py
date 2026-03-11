"""
extract_features.py — Extração de features de áudio via librosa.

Uso:
    python dsp/extract_features.py [--workers N] [--output PATH] [--balance-strategy S] [--labels L]

Lê os caminhos dos arquivos .wav do MongoDB, extrai features numéricas
e salva um DataFrame em features.parquet (uma linha por faixa).

Filtro de labels (LABELS no .env ou --labels):
    Vazio (padrão)  Processa todos os gêneros presentes no banco
    "a,b,c"         Processa apenas os labels listados (separados por vírgula)
    Exemplo: LABELS=metalcore,nu_metal,pop

Estratégias de balanceamento (BALANCE_STRATEGY no .env ou --balance-strategy):
    none         Processa todas as faixas (padrão)
    undersample  Limita todas as classes ao tamanho da menor classe
    balance      Limita classes acima da mediana ao valor da mediana;
                 classes abaixo da mediana ficam intactas

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
import random
from collections import defaultdict
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

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "features.parquet"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "50"))
BALANCE_STRATEGY = os.getenv("BALANCE_STRATEGY", "none")  # none | undersample | balance
LABELS = os.getenv("LABELS", "")  # csv de labels; vazio = todos os gêneros


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

def _parse_labels(raw: str) -> list[str] | None:
    """Converte 'a,b,c' em ['a','b','c']. Retorna None se vazio (= sem filtro)."""
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return items if items else None


def get_tracks(collection: Collection, labels: list[str] | None = None) -> list[dict]:
    """Retorna faixas do MongoDB que têm file_path registrado.

    Se labels for fornecido, filtra apenas os gêneros listados.
    """
    query: dict = {"file_path": {"$ne": ""}}
    if labels:
        query["label"] = {"$in": labels}
    return list(collection.find(
        query,
        {"_id": 0, "title": 1, "url": 1, "label": 1, "file_path": 1},
    ))


def _apply_balance_strategy(tracks: list[dict], strategy: str) -> list[dict]:
    """
    Filtra a lista de faixas conforme a estratégia de balanceamento.

    none        — retorna tudo sem alteração
    undersample — limita cada classe ao tamanho da menor classe (cap = min)
    balance     — limita classes acima da mediana ao valor da mediana;
                  classes abaixo da mediana ficam intactas
                  (reduz outliers sem penalizar as classes menores)
    """
    if strategy == "none":
        return tracks

    by_label: dict[str, list] = defaultdict(list)
    for t in tracks:
        by_label[t["label"]].append(t)

    counts = sorted(len(v) for v in by_label.values())

    if strategy == "undersample":
        cap = counts[0]
    elif strategy == "balance":
        n = len(counts)
        cap = counts[n // 2] if n % 2 == 1 else counts[n // 2 - 1]  # mediana
    else:
        raise ValueError(
            f"BALANCE_STRATEGY inválido: {strategy!r}. Use: none, undersample, balance"
        )

    logger.info("[BALANCE] Estratégia: %s | cap = %d faixas/classe", strategy, cap)
    result: list[dict] = []
    for label, items in sorted(by_label.items()):
        selected = random.sample(items, min(len(items), cap))
        logger.info("  %-15s %d → %d", label, len(items), len(selected))
        result.extend(selected)

    logger.info("  Total: %d → %d faixas selecionadas\n", len(tracks), len(result))
    return result


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _process_track(track: dict) -> dict | None:
    """Extrai features de uma faixa. Retorna None em caso de erro."""
    file_path = Path(track["file_path"])
    if not file_path.is_absolute():
        # Tenta resolver relativo à raiz do projeto; fallback para o diretório pai
        # (caso o ingest tenha sido rodado de fora da pasta music-classifier/)
        candidate = _PROJECT_ROOT / file_path
        if not candidate.exists():
            candidate = _PROJECT_ROOT.parent / file_path
        file_path = candidate
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


def _flush_checkpoint(
    rows: list, base_df: "pd.DataFrame | None", output_path: Path
) -> None:
    """
    Salva o parquet de forma atômica (escreve em .tmp e renomeia).
    Concatena base_df (linhas pré-existentes) com rows (novas nesta sessão).
    """
    new_df = pd.DataFrame(rows)
    final_df = pd.concat([base_df, new_df], ignore_index=True) if base_df is not None else new_df
    tmp = output_path.with_suffix(".tmp.parquet")
    final_df.to_parquet(tmp, index=False)
    tmp.replace(output_path)  # renomeio atômico — não corrompe o parquet se interrompido


def run_extraction(
    output_path: Path, max_workers: int, checkpoint_every: int,
    balance_strategy: str, labels: list[str] | None = None,
) -> None:
    """Pipeline principal: lê MongoDB → filtra labels → aplica balanceamento → extrai features → salva parquet."""
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    if labels:
        logger.info("[FILTER] Labels selecionados: %s", ", ".join(labels))
    else:
        logger.info("[FILTER] Nenhum filtro de label — processando todos os gêneros.")

    all_tracks = get_tracks(collection, labels)
    logger.info("[INFO] %d faixa(s) encontrada(s) no banco.", len(all_tracks))

    all_tracks = _apply_balance_strategy(all_tracks, balance_strategy)

    # Incremental: pula faixas já presentes no parquet existente
    if output_path.exists():
        base_df = pd.read_parquet(output_path)
        done_paths = set(base_df["file_path"].tolist())
        pending = [t for t in all_tracks if t["file_path"] not in done_paths]
        logger.info(
            "[INFO] %d já processada(s), %d na fila.",
            len(done_paths), len(pending),
        )
    else:
        base_df = None
        pending = all_tracks

    if not pending:
        logger.info("Nenhuma faixa nova para processar.")
        return

    logger.info(
        "\n[START] Extraindo features de %d faixa(s) com %d workers "
        "(checkpoint a cada %d)...\n",
        len(pending), max_workers, checkpoint_every,
    )

    rows: list = []
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

            # Checkpoint periódico
            if done > 0 and done % checkpoint_every == 0:
                _flush_checkpoint(rows, base_df, output_path)
                logger.info("  [CKPT] Checkpoint salvo (%d novas faixas acumuladas).", done)

    if not rows:
        logger.warning("Nenhuma feature extraída. Verifique os arquivos .wav.")
        return

    _flush_checkpoint(rows, base_df, output_path)
    final_rows = (len(base_df) if base_df is not None else 0) + len(rows)
    logger.info(
        "\nConcluído. %d extraída(s), %d erro(s). Parquet salvo em: %s",
        done, errors, output_path,
    )
    logger.info("Shape final: %d linhas × 372 colunas.", final_rows)


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
    parser.add_argument(
        "--checkpoint-every", "-c", type=int, default=CHECKPOINT_EVERY,
        help=f"Salvar parquet a cada N faixas extraídas (padrão: {CHECKPOINT_EVERY}, env: CHECKPOINT_EVERY)",
    )
    parser.add_argument(
        "--balance-strategy", "-b", default=BALANCE_STRATEGY,
        choices=["none", "undersample", "balance"],
        help=f"Estratégia de balanceamento de classes (padrão: {BALANCE_STRATEGY}, env: BALANCE_STRATEGY)",
    )
    parser.add_argument(
        "--labels", "-l", default=LABELS,
        help=(
            "Labels a incluir, separados por vírgula. "
            f"Vazio = todos os gêneros (padrão: '{LABELS}', env: LABELS). "
            "Exemplo: --labels metalcore,nu_metal,pop"
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_extraction(
        output_path=args.output,
        max_workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        balance_strategy=args.balance_strategy,
        labels=_parse_labels(args.labels),
    )
