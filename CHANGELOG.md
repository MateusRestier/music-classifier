# Changelog — Music Classifier

Formato: [MAJOR.MINOR] — descrição das mudanças agrupadas por etapa/sessão.

---

## [0.3] — 2026-03-11 — Checkpoint DSP + Backup/restore do MongoDB

### Adicionado
- `dsp/extract_features.py`: checkpoint periódico e escrita atômica
  - `_flush_checkpoint()` — escreve em `.tmp.parquet` e renomeia atomicamente
  - `CHECKPOINT_EVERY` (env + CLI `--checkpoint-every`) — salva a cada N faixas extraídas
- `ingest/mongo_crud.py`: comandos `dump` e `restore` via `mongodump`/`mongorestore`
  - `dump [--output DIR]` — salva o banco em `backups/dump_YYYYMMDD_HHMMSS/` por padrão
  - `restore <dump_dir> [--drop]` — restaura a partir de um dump; `--drop` evita duplicatas
  - `MONGODUMP_PATH` / `MONGORESTORE_PATH` configuráveis via env (default: `C:\Program Files\MongoDB\Tools\100\bin\`)
- `.env.example`: adicionadas variáveis `CHECKPOINT_EVERY`, `MONGODUMP_PATH`, `MONGORESTORE_PATH`
- `.gitignore`: adicionado `backups/`
- `CHANGELOG.md`: este arquivo

### Corrigido
- `ingest/mongo_crud.py` `dump`/`restore`: adicionado `--authenticationDatabase admin` — sem ele o mongodump falhava com `AuthenticationFailed` mesmo com URI e credenciais corretos ([troubleshooting #006](docs/troubleshooting.md))

---

## [0.2] — 2026-03-11 — Etapa 2: Extração de features (DSP)

### Adicionado
- `dsp/extract_features.py`: pipeline de extração de features de áudio via librosa
  - 369 features por faixa: MFCCs (40), Mel-Spectrogram (128), Chroma (12), Spectral Centroid, Spectral Rolloff, ZCR, RMS, Tempo (BPM)
  - Processamento paralelo via `ThreadPoolExecutor`
  - Modo incremental: pula faixas já presentes no parquet existente
  - Checkpoint periódico (configurável via `CHECKPOINT_EVERY`) com escrita atômica (`.tmp.parquet` → rename)
  - CLI: `--workers`, `--output`, `--checkpoint-every`
- `.env.example`: arquivo de configuração pronto para localhost (sem placeholders)
  - `MONGO_URI`, `FFMPEG_PATH`, `MAX_WORKERS`, `CHECKPOINT_EVERY`
- `docs/troubleshooting.md`: registro de problemas encontrados e soluções aplicadas
- `README.md`: adicionada Etapa 5 (extração de features) ao Início Rápido

### Corrigido
- `dsp/extract_features.py`: resolução de caminho com fallback duplo (`_PROJECT_ROOT / path` → `_PROJECT_ROOT.parent / path`) para compatibilidade com dataset criado fora da pasta do projeto
- `ingest/ingest.py`: `DATASET_ROOT` agora usa caminho absoluto relativo ao script (`Path(__file__).resolve().parent.parent / "dataset"`)

---

## [0.1] — 2026-03-11 — Etapa 1: Ingestão

### Adicionado
- `ingest/ingest.py`: download de áudio do YouTube via yt-dlp e armazenamento no MongoDB
  - Suporte a vídeos individuais e playlists no `CATALOG`
  - Deduplicação automática por URL
  - Conversão para `.wav` via ffmpeg
  - Configuração via variáveis de ambiente: `MONGO_URI`, `FFMPEG_PATH`, `MAX_WORKERS`
- `ingest/mongo_crud.py`: CLI para gerenciar a coleção de faixas
  - Comandos: `stats`, `list`, `search`, `get`, `add`, `update`, `delete`, `purge-broken`
- `docker-compose.yml`: MongoDB com autenticação (`admin:admin123`)
- `requirements.txt`: dependências do projeto
- `.gitignore`: `dataset/`, `logs/`, `*.wav`, `*.mp3`, `*.parquet`, `.env`, `__pycache__`

### Corrigido
- `ingest/ingest.py`: `UnicodeDecodeError` no output do subprocess (yt-dlp) — captura como bytes e decodifica com `errors="replace"` ([troubleshooting #001](docs/troubleshooting.md))
- `ingest/ingest.py`: playlists retornando 0 faixas — formato `%(id)s\t%(title)s` em vez de `%(.{id,title})s` ([troubleshooting #002](docs/troubleshooting.md))
- `ingest/ingest.py`: ffmpeg não encontrado pelo subprocess — passa `--ffmpeg-location FFMPEG_PATH` ([troubleshooting #003](docs/troubleshooting.md))
- `ingest/ingest.py`: títulos com acentos corrompidos — `PYTHONUTF8=1` no env do subprocess ([troubleshooting #004](docs/troubleshooting.md))

### Alterado
- `ingest/ingest.py`: substituído `print()` + `threading.Lock` por `logging` com dual handler (arquivo UTF-8 + stderr)
- `ingest/mongo_crud.py`: substituído `print()` por `logging`; adicionado `purge-broken` e `search`
- Todos os scripts: logs salvos em `logs/<script>_YYYYMMDD_HHMMSS.log` (gitignored)
