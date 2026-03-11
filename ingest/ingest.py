"""
ingest.py — Download de áudios do YouTube e inserção de metadados no MongoDB.

Uso:
    python ingest/ingest.py

Pré-requisitos:
    - ffmpeg instalado e no PATH
    - MongoDB rodando (docker compose up -d)
    - pip install -r requirements.txt
"""

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient
from pymongo.collection import Collection

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:admin123@localhost:27017/")
DB_NAME = "music_classifier"
COLLECTION_NAME = "tracks"

DATASET_ROOT = Path("dataset")

# Caminho explícito para o ffmpeg (necessário quando o WinGet Links não está no PATH do subprocess)
FFMPEG_PATH = os.getenv("FFMPEG_PATH", r"C:\Users\groun\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe")

# ---------------------------------------------------------------------------
# Catálogo de músicas a baixar
#
# Cada categoria aceita dois tipos de entrada (misturáveis):
#
#   Vídeo individual:
#     {"title": "Artista - Música", "url": "https://youtube.com/watch?v=..."}
#
#   Playlist inteira (título extraído automaticamente do YouTube):
#     {"playlist": "https://youtube.com/playlist?list=..."}
# ---------------------------------------------------------------------------

CATALOG: dict[str, list[dict]] = {
    "metalcore": [
        {"playlist": "https://www.youtube.com/playlist?list=PL7v1FHGMOadBTndBvtY4h213M10Pl9Y1c"},
        {"playlist": "https://www.youtube.com/playlist?list=PLhcGuOPZJV3yFcf6nXqTS22aq65yawRs1"},
    ],
    "nu_metal": [
        {"playlist": "https://www.youtube.com/playlist?list=PLf5aYiZPCNE0lnWZmC_pI6mtzyBS83Px1"},
    ],
    "sertanejo": [
        # Playlists com sucessos do Sertanejo e Sertanejo Universitário
        {"playlist": "https://www.youtube.com/playlist?list=PLCwAHfhr-Gc_wCv4yrPBiZFFyaRDMraMl"},
        {"playlist": "https://www.youtube.com/playlist?list=PL_Q15fKxrBb7QRCHi9uaeLOuAppg5Ebbg"},
    ],
    "pagode": [
        # Playlists focadas em rodas de Pagode e sucessos históricos
        {"playlist": "https://www.youtube.com/playlist?list=PLCwAHfhr-Gc_Ra3WPBZZDdzS1MNRgO1lq"},
        {"playlist": "https://www.youtube.com/playlist?list=PL_Q15fKxrBb5pckIW2RHwZbgf-FwRiCWr"},
    ],
    "pop": [
        # Playlists gigantescas com os maiores hits Pop globais
        {"playlist": "https://www.youtube.com/playlist?list=PLplXQ2cg9B_qrCVd1J_iId5SvP8Kf_BfS"},
    ]
}


# ---------------------------------------------------------------------------
# Funções
# ---------------------------------------------------------------------------

def get_collection() -> Collection:
    """Retorna a coleção MongoDB configurada."""
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]


def load_known_video_ids(collection: Collection) -> set[str]:
    """
    Retorna o conjunto de video_ids (ex: 'dQw4w9WgXcQ') já registrados no banco.
    Extrai o ID diretamente da URL salva, evitando um campo extra no schema.
    """
    known = set()
    for doc in collection.find({}, {"url": 1, "_id": 0}):
        vid_id = _extract_video_id(doc.get("url", ""))
        if vid_id:
            known.add(vid_id)
    return known


def _extract_video_id(url: str) -> str | None:
    """Extrai o video_id de uma URL do YouTube (watch?v= ou youtu.be/)."""
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return None


def build_output_path(label: str, title: str) -> Path:
    """Monta o caminho de saída: dataset/<label>/<titulo_sanitizado>.wav"""
    safe_title = "".join(c if c.isalnum() or c in " -_()" else "_" for c in title).strip()
    folder = DATASET_ROOT / label
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{safe_title}.wav"


def expand_playlist(playlist_url: str) -> list[dict]:
    """
    Usa yt-dlp --flat-playlist para listar todos os vídeos de uma playlist.
    Retorna lista de {"title": ..., "url": ...} sem fazer nenhum download.
    """
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "%(id)s\t%(title)s",  # tab-separated: id <TAB> título
        playlist_url,
    ]
    print(f"  [LIST] Expandindo playlist: {playlist_url}")
    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0:
        print(f"  [ERR]  Falha ao expandir playlist:\n{result.stderr.decode('utf-8', errors='replace')}")
        return []

    stdout = result.stdout.decode("utf-8", errors="replace")
    tracks = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        video_id, _, title = line.partition("\t")
        video_id = video_id.strip()
        title = title.strip() or video_id
        if video_id:
            tracks.append({
                "title": title,
                "url": f"https://www.youtube.com/watch?v={video_id}",
            })

    print(f"  [LIST] {len(tracks)} faixa(s) encontrada(s) na playlist.")
    return tracks


def download_audio(url: str, output_path: Path) -> bool:
    """
    Usa yt-dlp + ffmpeg para baixar o áudio em WAV sem compressão.
    Retorna True em caso de sucesso.
    """
    if output_path.exists():
        print(f"  [SKIP] Arquivo já existe: {output_path}")
        return True

    # yt-dlp baixa o melhor áudio disponível e ffmpeg converte para WAV PCM 16-bit
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--ffmpeg-location", FFMPEG_PATH,
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",          # melhor qualidade possível antes da conversão
        "--postprocessor-args", "ffmpeg:-acodec pcm_s16le",  # WAV sem compressão
        "--output", str(output_path.with_suffix("")),        # yt-dlp acrescenta .wav
        url,
    ]

    print(f"  [DOWN] {url}")
    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0:
        print(f"  [ERR]  yt-dlp falhou:\n{result.stderr.decode('utf-8', errors='replace')}")
        return False

    # yt-dlp pode gerar o arquivo com extensão diferente; normaliza para .wav
    generated = output_path.with_suffix("").with_suffix(".wav")
    if not generated.exists():
        # tenta encontrar qualquer arquivo gerado com mesmo stem
        candidates = list(output_path.parent.glob(f"{output_path.stem}*"))
        if candidates:
            candidates[0].rename(output_path)
        else:
            print(f"  [ERR]  Arquivo de saída não encontrado após download.")
            return False

    print(f"  [OK]   Salvo em: {output_path}")
    return True


def upsert_track(collection: Collection, title: str, url: str, label: str, file_path: Path) -> None:
    """
    Insere ou atualiza o documento da faixa no MongoDB.
    Chave de upsert: URL (evita duplicatas ao re-rodar o script).
    """
    doc = {
        "title": title,
        "url": url,
        "label": label,
        "file_path": str(file_path),
        "downloaded_at": datetime.now(timezone.utc),
    }
    collection.update_one({"url": url}, {"$set": doc}, upsert=True)
    print(f"  [DB]   Metadados salvos: {title!r} → {label}")


def resolve_tracks(entries: list[dict], seen_ids: set[str]) -> list[dict]:
    """
    Converte uma lista de entradas do catálogo em lista plana de {"title", "url"}.
    - Expande playlists via yt-dlp.
    - Deduplica por video_id dentro do batch atual (seen_ids é mutado in-place).
    """
    resolved = []
    for entry in entries:
        candidates = (
            expand_playlist(entry["playlist"])
            if "playlist" in entry
            else [{"title": entry["title"], "url": entry["url"]}]
        )
        for track in candidates:
            vid_id = _extract_video_id(track["url"])
            if vid_id in seen_ids:
                print(f"  [DUP]  Ignorado (já visto nesta sessão): {track['title']!r}")
                continue
            seen_ids.add(vid_id)
            resolved.append(track)
    return resolved


def run_ingestion(catalog: dict[str, list[dict]]) -> None:
    """Processa o catálogo completo: download + inserção no MongoDB."""
    collection = get_collection()

    if not any(catalog.values()):
        print("Catálogo vazio. Adicione entradas em CATALOG dentro de ingest.py.")
        return

    # IDs já persistidos no banco (execuções anteriores)
    known_ids = load_known_video_ids(collection)
    print(f"[INFO] {len(known_ids)} faixa(s) já registrada(s) no banco.\n")

    # IDs vistos nesta sessão (evita duplicatas entre playlists do mesmo run)
    seen_ids: set[str] = set(known_ids)

    for label, entries in catalog.items():
        if not entries:
            continue

        tracks = resolve_tracks(entries, seen_ids)

        if not tracks:
            print(f"\n=== Categoria: {label} — nenhuma faixa nova. ===")
            continue

        print(f"\n=== Categoria: {label} ({len(tracks)} faixa(s) nova(s)) ===")

        for track in tracks:
            title = track["title"]
            url = track["url"]
            output_path = build_output_path(label, title)

            success = download_audio(url, output_path)
            if success:
                upsert_track(collection, title, url, label, output_path)
            print()

    print("Ingestão concluída.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_ingestion(CATALOG)
