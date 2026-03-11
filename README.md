# music-classifier

Pipeline ponta a ponta para classificação e clusterização de gêneros musicais.

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

> **Atenção (Windows):** O yt-dlp pode não encontrar o ffmpeg instalado via WinGet mesmo que ele esteja no PATH do terminal. Nesse caso, defina o caminho explicitamente antes de rodar o script:
> ```powershell
> $env:FFMPEG_PATH = "C:\Users\<seu_usuario>\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
> python ingest/ingest.py
> ```
> O caminho padrão hardcoded no script é `C:\Users\groun\...` — ajuste em `ingest/ingest.py` na variável `FFMPEG_PATH` se necessário.

### 2. Instalar dependências Python

```bash
pip install -r requirements.txt
```

### 3. Subir o MongoDB

```bash
docker compose up -d
```

Conexão: `mongodb://admin:admin123@localhost:27017/`

### 4. Configurar e rodar a ingestão

Abra [ingest/ingest.py](ingest/ingest.py) e preencha o dicionário `CATALOG`. Cada categoria aceita **vídeos individuais**, **playlists inteiras** ou ambos misturados:

```python
CATALOG = {
    "metalcore": [
        # Playlist inteira — títulos extraídos automaticamente do YouTube
        {"playlist": "https://www.youtube.com/playlist?list=..."},
    ],
    "nu_metal": [
        # Vídeo individual — título definido manualmente
        {"title": "Linkin Park - In The End", "url": "https://www.youtube.com/watch?v=..."},
        # Pode misturar playlist e vídeos individuais na mesma categoria
        {"playlist": "https://www.youtube.com/playlist?list=..."},
    ],
}
```

O script deduplica automaticamente: a mesma música não será baixada duas vezes, mesmo que apareça em playlists diferentes ou em execuções anteriores.

Depois execute:

```bash
python ingest/ingest.py
```

Os arquivos `.wav` serão salvos em `dataset/<label>/` e os metadados no MongoDB.

### 5. Extrair features de áudio

Com o dataset ingerido, execute a extração de features DSP:

```bash
python dsp/extract_features.py
```

O script lê os caminhos do MongoDB, processa cada `.wav` em paralelo e salva `features.parquet` na raiz do projeto.

**Opções:**
```bash
python dsp/extract_features.py --workers 6   # aumentar paralelismo (padrão: 4)
python dsp/extract_features.py --output outro/caminho.parquet
```

A extração é **incremental**: se `features.parquet` já existir, apenas as faixas novas são processadas.

---

## Estrutura do Projeto

```
music-classifier/
├── dataset/                  # áudios .wav (gitignored)
├── logs/                     # logs dos scripts (gitignored)
├── docs/
│   └── visao_do_projeto.md   # arquitetura e roadmap
├── ingest/
│   ├── ingest.py             # download + MongoDB
│   └── mongo_crud.py         # CLI CRUD do banco
├── dsp/
│   └── extract_features.py   # extração de features via librosa
├── modeling/                 # ML (Etapa 3)
├── app/                      # Streamlit (Etapa 4)
├── docker-compose.yml
├── requirements.txt
└── README.md
```
