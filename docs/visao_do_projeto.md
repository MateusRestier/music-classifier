# Visão do Projeto — Music Classifier

## Objetivo

Construir um pipeline ponta a ponta para **classificar e clusterizar gêneros e subgêneros musicais** (ex: nu metal, metalcore, sertanejo, pagode, pop).

---

## Arquitetura Geral

```
YouTube URLs
     │
     ▼
┌─────────────────────┐
│  Ingestão (yt-dlp)  │  ──►  /dataset/<label>/<titulo>.wav  (disco local)
└─────────────────────┘
     │ metadados
     ▼
┌─────────────────────┐
│  MongoDB (Docker)   │  coleta: título, URL, label, caminho
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  DSP (librosa)      │  extrai: Mel-Spectrogram, MFCCs, BPM, chroma/harmonia
└─────────────────────┘
     │ feature vectors
     ▼
┌─────────────────────────────────────────┐
│  Modelagem ML                           │
│   • Clusterização  → K-Means / HDBSCAN  │
│   • Classificação  → Random Forest /    │
│                       XGBoost           │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────┐
│  App Streamlit      │  upload .mp3/.wav → espectrograma + predição de gênero
└─────────────────────┘
```

---

## Etapas Detalhadas

### Etapa 1 — Ingestão e Base de Dados Estática

**Objetivo:** Montar o dataset offline com músicas reais e músicas geradas por IA.

- Ferramenta de download: `yt-dlp` + `ffmpeg` (saída `.wav`, sem compressão)
- Estrutura de diretórios:
  ```
  dataset/
  ├── metalcore_real/
  ├── nu_metal_real/
  ├── sertanejo_real/
  ├── pagode_real/
  ├── pop_real/
  ├── metalcore_ia/
  └── ...
  ```
- Metadados persistidos no MongoDB (coleção `tracks`):
  ```json
  {
    "title":      "Nome da faixa",
    "url":        "https://youtube.com/watch?v=...",
    "label":      "metalcore_real",
    "file_path":  "dataset/metalcore_real/nome.wav",
    "downloaded_at": "2026-03-11T00:00:00"
  }
  ```

**Entregáveis:** `ingest/ingest.py`, `docker-compose.yml`

---

### Etapa 2 — Processamento de Sinais (DSP)

**Objetivo:** Converter cada `.wav` num vetor de features numéricas.

| Feature | Biblioteca | Descrição |
|---|---|---|
| Mel-Spectrogram | librosa | Representação tempo-frequência perceptual |
| MFCCs (13–40 coef.) | librosa | "Impressão digital" tímbrica do som |
| BPM / Tempo | librosa.beat | Velocidade rítmica |
| Chroma / Harmonia | librosa.feature | Tonalidade e progressão harmônica |
| Spectral Centroid | librosa.feature | "Brilho" do som |
| Zero-Crossing Rate | librosa.feature | Indicador de percussão/ruído |

Saída: DataFrame `features.parquet` com uma linha por faixa.

**Entregáveis:** `dsp/extract_features.py`

---

### Etapa 3 — Modelagem

#### 3a. Clusterização (exploratória)
- Redução de dimensionalidade: PCA → t-SNE / UMAP
- Algoritmo: K-Means (k definido por Elbow + Silhouette Score) ou HDBSCAN
- Validação visual: scatter plot colorido por cluster vs. label real

#### 3b. Classificação supervisionada
- Modelos: Random Forest, XGBoost
- Divisão: 80/20 treino/teste, stratified split
- Métricas: Accuracy, F1-macro, Confusion Matrix
- Exportação do modelo: `joblib` → `models/classifier.pkl`

**Entregáveis:** `modeling/cluster.py`, `modeling/classify.py`

---

### Etapa 4 — Produto Final (Streamlit)

**Funcionalidades:**
1. Upload de arquivo `.mp3` ou `.wav`
2. Extração de features em tempo real (mesmo pipeline da Etapa 2)
3. Exibição do Mel-Spectrogram interativo (matplotlib / plotly)
4. Predição: gênero + probabilidade por classe

**Entregáveis:** `app/app.py`

---

## Estrutura de Pastas do Repositório

```
music-classifier/
├── dataset/                  # áudios .wav (gitignored)
├── docs/
│   └── visao_do_projeto.md
├── ingest/
│   └── ingest.py             # download + inserção MongoDB
├── dsp/
│   └── extract_features.py   # extração de features via librosa
├── modeling/
│   ├── cluster.py
│   └── classify.py
├── app/
│   └── app.py                # Streamlit
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Download de áudio | yt-dlp + ffmpeg |
| Banco de metadados | MongoDB (Docker) |
| DSP | librosa, numpy, scipy |
| ML | scikit-learn, xgboost |
| Visualização | matplotlib, plotly |
| App | Streamlit |
| Infra local | Docker Compose |

---

## Decisões de Design

- **WAV sem compressão** garante fidelidade total para DSP (evita artefatos de codec MP3/OGG).
- **MongoDB** foi escolhido pelo schema flexível — metadados de vídeos do YouTube variam muito.
- **Dataset estático offline** antes de qualquer modelo evita retrabalho e garante reprodutibilidade.
- **Labels manuais** na Etapa 1 permitem tanto clusterização não-supervisionada (para explorar se o modelo "aprende" os gêneros sozinho) quanto classificação supervisionada por gênero.
