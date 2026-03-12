# Visão do Projeto — SoundDNA

## Objetivo

Construir um pipeline ponta a ponta para **classificar e clusterizar gêneros e subgêneros musicais** a partir de áudio bruto, cobrindo estilos brasileiros e internacionais.

---

## Arquitetura Geral

```
YouTube URLs
     │
     ▼
┌─────────────────────┐
│  Ingestão (yt-dlp)  │  ──►  dataset/<label>/<titulo>.wav  (disco local)
└─────────────────────┘
     │ metadados
     ▼
┌─────────────────────┐
│  MongoDB (Docker)   │  título, URL, label, file_path, downloaded_at
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  DSP (librosa)      │  369 features: MFCCs, Mel-Spectrogram, Chroma,
└─────────────────────┘  Spectral Centroid/Rolloff, ZCR, RMS, BPM
     │ features.parquet
     ▼
┌─────────────────────────────────────────┐
│  Modelagem ML                           │
│   • Clusterização  → K-Means / t-SNE    │
│   • Classificação  → Random Forest /    │
│                       XGBoost           │
└─────────────────────────────────────────┘
     │ models/*.pkl
     ▼
┌──────────────────────────────────────────────────────┐
│  App Streamlit                                       │
│  upload .mp3/.wav ou link YouTube →                  │
│  análise espectral + BPM + tonalidade + predição     │
└──────────────────────────────────────────────────────┘
```

---

## Etapas Detalhadas

### Etapa 1 — Ingestão e Base de Dados

**Objetivo:** Montar o dataset com músicas reais do YouTube, organizadas por gênero.

- Ferramenta de download: `yt-dlp` + `ffmpeg` (saída `.wav` PCM 16-bit, sem compressão)
- Estrutura de diretórios: `dataset/<label>/<titulo>.wav`
- Metadados persistidos no MongoDB (coleção `tracks`):
  ```json
  {
    "title":        "Nome da faixa",
    "url":          "https://youtube.com/watch?v=...",
    "label":        "metalcore",
    "file_path":    "dataset/metalcore/nome.wav",
    "downloaded_at": "2026-03-12T00:00:00Z"
  }
  ```
- `CATALOG` em `ingest/ingest.py`: dicionário `{label: [playlists/vídeos]}`
- Deduplicação automática por `video_id` — mesma música nunca é baixada duas vezes

**Gêneros no CATALOG (22):**

| Família | Labels |
|---|---|
| Metal/Rock | `metalcore`, `nu_metal`, `alternative_rock`, `heavy_metal`, `punk_rock` |
| Brasileiro | `sertanejo`, `pagode`, `samba`, `funk`, `mpb`, `forro`, `axe` |
| Pop | `pop`, `kpop` |
| Eletrônico | `edm`, `lo-fi`, `trap` |
| Clássico/Erudito | `classica`, `opera`, `jazz` |
| Global | `reggae`, `rap` |

**Entregáveis:** `ingest/ingest.py`, `ingest/mongo_crud.py`, `docker-compose.yml`

---

### Etapa 2 — Processamento de Sinais (DSP)

**Objetivo:** Converter cada `.wav` num vetor de 369 features numéricas.

| Feature | Dim | Biblioteca | Descrição |
|---|---|---|---|
| MFCCs (40 coef.) | 80 | librosa | Timbre — mean + std por coeficiente |
| Mel-Spectrogram (128 bins) | 256 | librosa | Representação tempo-frequência perceptual |
| Chroma (12 bins) | 24 | librosa | Conteúdo harmônico e tonalidade |
| Spectral Centroid | 2 | librosa | "Brilho" do som |
| Spectral Rolloff | 2 | librosa | Frequência de corte a 85% da energia |
| Zero-Crossing Rate | 2 | librosa | Indicador de percussão/ruído |
| RMS Energy | 2 | librosa | Energia do sinal |
| Tempo (BPM) | 1 | librosa.beat | Velocidade rítmica |

- Segmento analisado: 30s a partir de t=30s (evita intros)
- Processamento paralelo via `ThreadPoolExecutor`
- Modo incremental: pula faixas já presentes no parquet
- Checkpoint periódico com escrita atômica (`.tmp.parquet` → rename)
- Estratégias de balanceamento: `none`, `undersample`, `balance` (mediana)
- Filtro por gênero via `--labels` ou env `LABELS`

**Entregáveis:** `dsp/extract_features.py` → `features.parquet`

---

### Etapa 3 — Modelagem

#### 3a. Clusterização (exploratória)
- Pipeline: StandardScaler → PCA (95% variância) → K-Means (Elbow + Silhouette k=2..10) → t-SNE
- Plots salvos em `modeling/plots/`
- Permite visualizar se as features conseguem separar os gêneros sem supervisão

#### 3b. Classificação supervisionada
- Pipeline: StandardScaler → split 80/20 estratificado → Random Forest + XGBoost
- Métricas: Accuracy, F1-macro, Confusion Matrix por classe
- Exporta o melhor modelo por F1-macro: `models/classifier.pkl`, `scaler.pkl`, `label_encoder.pkl`
- Resultado com 5 gêneros (1.102 faixas, sem balanceamento):
  - XGBoost: F1-macro = **0.7256** ✓
  - Random Forest: F1-macro = 0.6624
  - Fraqueza conhecida: metalcore recall=36% (confusão com nu_metal — gêneros acusticamente próximos)

**Entregáveis:** `modeling/cluster.py`, `modeling/classify.py`

---

### Etapa 4 — App Streamlit ✅

**Status:** Concluído.

**Funcionalidades:**

1. **Duas fontes de entrada:**
   - Upload de arquivo `.mp3` ou `.wav`
   - Link de vídeo do YouTube (yt-dlp baixa WAV temporário automaticamente)

2. **Métricas rápidas:** duração do segmento, BPM estimado, tonalidade (chroma dominante), energia RMS média

3. **5 abas de análise de sinal:**
   - **Forma de onda** — amplitude × tempo com beats detectados marcados (Plotly, interativo)
   - **Mel-Spectrogram** — tempo × frequência Mel em dB, colormap magma
   - **MFCCs** — heatmap 40 coeficientes × tempo
   - **Chroma** — energia por classe de pitch (C…B) × tempo
   - **Features espectrais** — centróide, rolloff, ZCR e RMS quadro a quadro (Plotly, 4 subplots)

4. **Predição de gênero:**
   - Métricas: 1º e 2º gênero mais prováveis com probabilidade
   - Bar chart ordenado por probabilidade
   - Radar chart da distribuição completa por classe

**Entregáveis:** `app/app.py`

---

## Estrutura de Pastas

```
sound-dna/
├── dataset/                  # áudios .wav (gitignored)
├── models/                   # classifier.pkl, scaler.pkl, label_encoder.pkl (gitignored)
├── logs/                     # logs dos scripts (gitignored)
├── docs/
│   ├── visao_do_projeto.md
│   └── troubleshooting.md
├── ingest/
│   ├── ingest.py             # download + inserção MongoDB
│   └── mongo_crud.py         # CLI: stats/list/search/add/update/delete/purge-broken/dump/restore
├── dsp/
│   └── extract_features.py   # extração de 369 features via librosa
├── modeling/
│   ├── cluster.py            # clusterização exploratória
│   └── classify.py           # classificação supervisionada
├── app/
│   └── app.py                # Streamlit
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── CHANGELOG.md
```

---

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Download de áudio | yt-dlp + ffmpeg |
| Banco de metadados | MongoDB (Docker) |
| DSP | librosa, numpy, scipy |
| ML | scikit-learn, xgboost, joblib |
| Visualização | matplotlib, plotly, seaborn |
| App | Streamlit |
| Infra local | Docker Compose |

---

## Decisões de Design

- **WAV sem compressão** garante fidelidade total para DSP (evita artefatos de codec MP3/OGG).
- **MongoDB** foi escolhido pelo schema flexível — metadados de vídeos do YouTube variam muito.
- **Segmento t=30s–60s** evita intros instrumentais e aumenta consistência entre faixas.
- **369 features** combinam informação tímbrica (MFCCs), harmônica (Chroma), rítmica (BPM) e energética (RMS, ZCR) para cobrir as principais dimensões que distinguem gêneros.
- **Labels sem acentos e sem `/`** evitam problemas de filesystem no Windows (`forro` em vez de `forró`, `edm` em vez de `Electronic/EDM`).
- **DB_NAME = "music_classifier"** mantido para preservar compatibilidade com o banco MongoDB existente; não precisa coincidir com o nome do repositório.
