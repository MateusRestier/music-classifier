# SoundDNA

Pipeline ponta a ponta para classificação e clusterização de gêneros musicais — da ingestão via YouTube até um app Streamlit com análise espectral e predição por ML.

Veja [docs/visao_do_projeto.md](docs/visao_do_projeto.md) para a arquitetura completa.

---

## Início Rápido

### 1. Pré-requisitos

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e rodando
- [ffmpeg](https://ffmpeg.org/download.html) instalado

**Instalar o ffmpeg no Windows:**
```powershell
winget install --id Gyan.FFmpeg -e
```

> **Atenção (Windows):** O yt-dlp pode não encontrar o ffmpeg instalado via WinGet mesmo que ele esteja no PATH do terminal. Nesse caso, defina o caminho no `.env`:
> ```
> FFMPEG_PATH=C:\Users\<seu_usuario>\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe
> ```

### 2. Instalar dependências Python

```bash
pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente

Copie o arquivo de exemplo e ajuste conforme necessário:

```bash
cp .env.example .env
```

As variáveis mais importantes estão descritas em `.env.example`.

### 4. Subir o MongoDB

```bash
docker compose up -d
```

Conexão: `mongodb://admin:admin123@localhost:27017/`

### 5. Configurar e rodar a ingestão

Abra [ingest/ingest.py](ingest/ingest.py) e preencha o dicionário `CATALOG`. Cada gênero aceita **vídeos individuais**, **playlists inteiras** ou ambos misturados:

```python
CATALOG = {
    "metalcore": [
        {"playlist": "https://www.youtube.com/playlist?list=..."},
    ],
    "pagode": [
        {"title": "Artista - Música", "url": "https://www.youtube.com/watch?v=..."},
        {"playlist": "https://www.youtube.com/playlist?list=..."},
    ],
}
```

O script deduplica automaticamente: a mesma música não é baixada duas vezes, mesmo em execuções repetidas.

```bash
python ingest/ingest.py
```

Os `.wav` são salvos em `dataset/<label>/` e os metadados no MongoDB.

**Gerenciar o banco:**
```bash
python ingest/mongo_crud.py stats                        # contagem por gênero
python ingest/mongo_crud.py list --label metalcore       # listar faixas
python ingest/mongo_crud.py search "Linkin Park"         # buscar por título
python ingest/mongo_crud.py purge-broken                 # remover entradas sem arquivo
python ingest/mongo_crud.py dump                         # backup completo
python ingest/mongo_crud.py restore <dump_dir> --drop    # restaurar
```

### 6. Extrair features de áudio (DSP)

```bash
python dsp/extract_features.py
```

Lê os caminhos do MongoDB, processa cada `.wav` em paralelo e salva `features.parquet` na raiz. A extração é **incremental**: só processa faixas novas.

**Opções úteis:**
```bash
python dsp/extract_features.py --workers 6                        # paralelismo
python dsp/extract_features.py --labels metalcore,nu_metal        # filtrar gêneros
python dsp/extract_features.py --balance-strategy balance         # balancear classes
python dsp/extract_features.py --output outro/caminho.parquet
```

Estratégias de balanceamento: `none` (padrão), `undersample` (cap = min), `balance` (cap = mediana).

### 7. Treinar o modelo (Modelagem)

```bash
python modeling/cluster.py    # clusterização exploratória (PCA → K-Means → t-SNE)
python modeling/classify.py   # classificação supervisionada (Random Forest + XGBoost)
```

O melhor modelo é exportado em `models/` (`classifier.pkl`, `scaler.pkl`, `label_encoder.pkl`).

### 8. Rodar o app

```bash
streamlit run app/app.py
```

O app permite analisar músicas via **upload de arquivo** (`.mp3`/`.wav`) ou **link do YouTube**, e exibe:
- Métricas rápidas: BPM, tonalidade, energia RMS
- 5 visualizações de sinal: forma de onda, Mel-Spectrogram, MFCCs, Chroma, features espectrais
- Predição de gênero com bar chart e radar chart de probabilidades

---

## Estrutura do Projeto

```
sound-dna/
├── dataset/                  # áudios .wav (gitignored)
├── models/                   # modelos treinados .pkl (gitignored)
├── logs/                     # logs dos scripts (gitignored)
├── docs/
│   ├── visao_do_projeto.md   # arquitetura e decisões de design
│   └── troubleshooting.md    # problemas encontrados e soluções
├── ingest/
│   ├── ingest.py             # download yt-dlp → MongoDB
│   └── mongo_crud.py         # CLI: stats/list/search/add/delete/dump/restore
├── dsp/
│   └── extract_features.py   # 369 features/faixa via librosa → features.parquet
├── modeling/
│   ├── cluster.py            # PCA → K-Means → t-SNE
│   └── classify.py           # Random Forest + XGBoost → exporta melhor modelo
├── app/
│   └── app.py                # Streamlit: análise espectral + predição de gênero
├── docker-compose.yml        # MongoDB (admin:admin123)
├── requirements.txt
├── .env.example
└── CHANGELOG.md
```
