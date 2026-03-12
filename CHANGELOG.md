# Changelog — SoundDNA

Formato: [MAJOR.MINOR] — descrição das mudanças agrupadas por etapa/sessão.

---

## [1.0] — 2026-03-12 — Suporte a .env via python-dotenv

### Adicionado
- `requirements.txt`: `python-dotenv>=1.0.0`
- `dsp/extract_features.py`, `ingest/ingest.py`, `ingest/mongo_crud.py`, `app/app.py`: `load_dotenv()` no topo — variáveis do `.env` (como `BALANCE_STRATEGY`) agora são lidas automaticamente sem precisar exportar no shell

---

## [0.9] — 2026-03-12 — Revisão da documentação

### Alterado
- `README.md`: reescrito completo — pipeline de 8 etapas (inclui Streamlit), comandos do `mongo_crud.py`, opções de `extract_features.py`, estrutura de pastas atualizada (`sound-dna/`, `models/`, `docs/troubleshooting.md`)
- `docs/visao_do_projeto.md`: atualizado para refletir o estado atual do projeto
  - Etapa 1: removidos sufixos `_real`/`_ia` (nunca usados); tabela com os 22 gêneros do CATALOG
  - Etapa 2: tabela de features com dimensionalidade explícita por grupo (total 369)
  - Etapa 3: resultado real documentado (XGBoost F1=0.7256, fraqueza metalcore recall=36%)
  - Etapa 4: marcada como ✅ concluída com funcionalidades reais descritas
  - Decisões de Design: rationale dos labels sem acento e do `DB_NAME` mantido

---

## [0.8] — 2026-03-12 — Expansão do CATALOG (22 gêneros)

### Adicionado
- `ingest/ingest.py`: CATALOG expandido de 5 para 22 gêneros com playlists preenchidas
  - Novos gêneros: `alternative_rock`, `heavy_metal`, `punk_rock`, `samba`, `kpop`, `funk`, `mpb`, `classica`, `opera`, `edm`, `forro`, `axe`, `jazz`, `lo-fi`, `reggae`, `rap`, `trap`

### Alterado
- Família rock reestruturada por clusters acústicos: `post_hardcore` e `progressive_metal` removidos (absorvidos por `metalcore` e `heavy_metal` respectivamente); `alternative_rock` adicionado
- Labels normalizados para snake_case sem acentos e sem caracteres especiais:
  - `"clássica"` → `"classica"`, `"ópera"` → `"opera"`, `"forró"` → `"forro"`, `"axé"` → `"axe"`
  - `"Electronic/EDM"` → `"edm"` *(bug: `/` criaria subdiretório no filesystem)*
  - `"funk_carioca"` → `"funk"`

---

## [0.7] — 2026-03-12 — Rename para SoundDNA

### Alterado
- Projeto renomeado de `music-classifier` para **SoundDNA** (repositório GitHub: `sound-dna`)
- Atualizado nome em: `README.md`, `CHANGELOG.md`, `docs/visao_do_projeto.md`, `docs/troubleshooting.md`, `app/app.py` (`page_title` + `st.title`), `dsp/extract_features.py` (comentário + CLI description), `ingest/mongo_crud.py` (CLI description)
- `DB_NAME = "music_classifier"` mantido para preservar compatibilidade com o banco MongoDB existente

---

## [0.6] — 2026-03-12 — Etapa 4: App Streamlit (SoundDNA)

### Adicionado
- `app/app.py`: aplicação Streamlit completa com duas abas de entrada
  - **Aba "Upload"**: aceita `.mp3` ou `.wav`; player de áudio nativo do Streamlit
  - **Aba "YouTube"**: cola URL de vídeo → yt-dlp baixa para WAV temporário → mesmo pipeline
  - Validação de URL YouTube via regex; mensagem de erro amigável; `shutil.rmtree` no `finally`
- Métricas rápidas (4 colunas): duração do segmento, BPM estimado, tonalidade (chroma dominante), energia RMS média
- 5 abas de análise de sinal:
  - **Forma de onda**: amplitude × tempo com beats marcados (Plotly, interativo; downsample automático a 8 000 pts)
  - **Mel-Spectrogram**: tempo × frequência Mel em dB, colormap `magma` (matplotlib + librosa)
  - **MFCCs**: heatmap 40 coeficientes × tempo, colormap `coolwarm` (matplotlib + librosa)
  - **Chroma**: energia por classe de pitch (C…B) × tempo, colormap `YlOrRd` (matplotlib + librosa)
  - **Features espectrais**: 4 subplots Plotly interativos — centróide, rolloff, ZCR e RMS quadro a quadro
- Predição de gênero (3 colunas): métricas de 1º e 2º mais prováveis + bar chart + radar chart (Plotly)
- `_compute_analysis(y, sr)`: todas as matrizes e features computadas uma única vez e reutilizadas
- `load_models()`: `@st.cache_resource` — modelos carregados uma única vez por sessão
- Layout `wide` para melhor aproveitamento do espaço dos gráficos

---

## [0.5] — 2026-03-11 — Filtro de labels na extração de features

### Adicionado
- `dsp/extract_features.py`: variável `LABELS` para filtrar gêneros incluídos no parquet
  - Vazio (padrão) = todos os gêneros do banco
  - Lista CSV: `LABELS=metalcore,nu_metal,pop` — inclui apenas os listados
  - Filtragem feita diretamente na query do MongoDB (`$in`)
  - Configurável via env `LABELS` ou CLI `--labels` / `-l`
  - Helper `_parse_labels(raw)`: faz strip dos espaços, retorna `None` se vazio (= sem filtro)
- `.env.example`: adicionada variável `LABELS=` com comentários explicativos

---

## [0.4] — 2026-03-11 — Etapa 3: Modelagem + Balanceamento de classes

### Adicionado
- `modeling/cluster.py`: clusterização exploratória
  - Pipeline: StandardScaler → PCA (95% variância) → K-Means (Elbow + Silhouette k=2..10) → t-SNE
  - Plots salvos em `modeling/plots/`: `elbow_silhouette.png`, `tsne_k{N}.png`
  - CLI: `--input`, `--k` (força k), `--output-dir`
- `modeling/classify.py`: classificação supervisionada
  - Pipeline: StandardScaler → split 80/20 estratificado → Random Forest + XGBoost
  - Exporta o melhor modelo (por F1-macro) em `models/classifier.pkl`, `scaler.pkl`, `label_encoder.pkl`
  - Plots: `confusion_matrix_*.png`, `feature_importance_rf.png`
  - Resultado inicial: XGBoost F1-macro=0.7256 vs RF F1-macro=0.6624
- `dsp/extract_features.py`: estratégia de balanceamento de classes (`BALANCE_STRATEGY`)
  - `none` — processa todas as faixas (padrão)
  - `undersample` — limita todas as classes ao tamanho da menor (cap = min)
  - `balance` — limita classes acima da mediana ao valor da mediana; classes menores ficam intactas
  - Configurável via env `BALANCE_STRATEGY` ou CLI `--balance-strategy` / `-b`
- `requirements.txt`: adicionado `seaborn>=0.13.0`
- `.env.example`: adicionada variável `BALANCE_STRATEGY=none` com comentários explicativos
- `.gitignore`: adicionado `modeling/plots/`

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
