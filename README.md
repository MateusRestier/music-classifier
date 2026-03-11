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

Abra [ingest/ingest.py](ingest/ingest.py) e preencha o dicionário `CATALOG` com seus títulos e URLs do YouTube:

```python
CATALOG = {
    "metalcore": [
        {"title": "Architects - Doomsday", "url": "https://www.youtube.com/watch?v=..."},
    ],
    "nu_metal": [
        {"title": "Linkin Park - In The End", "url": "https://www.youtube.com/watch?v=..."},
    ],
    "sertanejo": [
        {"title": "Exemplo - Nome da Música", "url": "https://www.youtube.com/watch?v=..."},
    ],
}
```

Depois execute:

```bash
python ingest/ingest.py
```

Os arquivos `.wav` serão salvos em `dataset/<label>/` e os metadados no MongoDB.

---

## Estrutura do Projeto

```
music-classifier/
├── dataset/                  # áudios .wav (gitignored)
├── docs/
│   └── visao_do_projeto.md   # arquitetura e roadmap
├── ingest/
│   └── ingest.py             # download + MongoDB
├── dsp/                      # extração de features (Etapa 2)
├── modeling/                 # ML (Etapa 3)
├── app/                      # Streamlit (Etapa 4)
├── docker-compose.yml
├── requirements.txt
└── README.md
```
