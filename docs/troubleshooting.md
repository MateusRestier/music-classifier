# Troubleshooting — SoundDNA

Registro de problemas encontrados durante o desenvolvimento, com causa raiz e solução aplicada.

---

## [001] UnicodeDecodeError no output do subprocess (yt-dlp)

**Data:** Etapa 1 — Ingestão
**Arquivo:** `ingest/ingest.py`

### Sintoma
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x92 in position ...
AttributeError: 'NoneType' object has no attribute 'splitlines'
```

### Causa
`subprocess.run(..., text=True, encoding="utf-8")` faz com que `stdout` seja decodificado automaticamente. No Windows, o yt-dlp (processo Python filho) usa o encoding do sistema (CP1252) para escrever na saída. Caracteres especiais como `'` (0x92 em CP1252) são inválidos em UTF-8, causando exceção. Quando a decodificação falha, `result.stdout` retorna `None`.

### Solução
Remover `text=True` e `encoding=` do `subprocess.run`, capturando a saída como bytes e decodificando manualmente com `errors="replace"`:

```python
# Antes
result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
stdout = result.stdout.splitlines()

# Depois
result = subprocess.run(cmd, capture_output=True)
stdout = result.stdout.decode("utf-8", errors="replace")
```

---

## [002] Playlists retornando 0 faixas

**Data:** Etapa 1 — Ingestão
**Arquivo:** `ingest/ingest.py` → `expand_playlist()`

### Sintoma
```
[LIST] 0 faixa(s) encontrada(s) na playlist.
```
Todas as playlists retornavam zero faixas, mesmo com URLs válidas.

### Causa
O formato `--print "%(.{id,title})s"` do yt-dlp gera saída no estilo de dict Python com aspas simples (ex: `{'id': 'abc', 'title': "Música"}`). O `json.loads()` falhava silenciosamente em cada linha porque aspas simples não são JSON válido.

### Solução
Trocar o formato para tab-separated e parsear com `str.partition("\t")`:

```python
# Antes
cmd = ["yt-dlp", "--flat-playlist", "--print", "%(.{id,title})s", ...]
# json.loads() falhava em cada linha

# Depois
cmd = ["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s", ...]
video_id, _, title = line.partition("\t")
```

---

## [003] ffmpeg não encontrado pelo subprocess

**Data:** Etapa 1 — Ingestão
**Arquivo:** `ingest/ingest.py`

### Sintoma
yt-dlp retornava erro de conversão indicando que o ffmpeg não estava disponível, mesmo com `ffmpeg` funcionando normalmente no terminal.

### Causa
O ffmpeg foi instalado via WinGet, cujos executáveis ficam em `C:\Users\<user>\AppData\Local\Microsoft\WinGet\Links\`. Esse diretório está no PATH do terminal interativo, mas **não é herdado pelos processos filhos** criados via `subprocess.run()`.

### Solução
Adicionar a constante `FFMPEG_PATH` apontando para o executável, e passá-la ao yt-dlp via `--ffmpeg-location`. O caminho é configurável via variável de ambiente:

```python
FFMPEG_PATH = os.getenv("FFMPEG_PATH", r"C:\Users\groun\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe")

cmd = ["yt-dlp", "--ffmpeg-location", FFMPEG_PATH, ...]
```

**Override via PowerShell (se necessário):**
```powershell
$env:FFMPEG_PATH = "C:\Users\<seu_usuario>\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
```

---

## [004] Títulos de músicas salvos com caracteres corrompidos

**Data:** Etapa 1 — Ingestão
**Arquivo:** `ingest/ingest.py` → `expand_playlist()`

### Sintoma
Títulos com acentos salvos incorretamente no MongoDB e no nome dos arquivos:
```
'Grupo Chocolate, Turma do Pagode - Al? Virginia (Ao Vivo)' → pagode
```
O caractere `?` (U+FFFD — Unicode replacement character) substituía letras acentuadas como `ô`, `í`, `ã`.

### Causa
O yt-dlp é um processo Python que usa o encoding do sistema para I/O (`stdout`). No Windows, o encoding padrão é CP1252. Caracteres como `ô` (0xF4 em CP1252) são bytes inválidos quando interpretados como UTF-8, então `errors="replace"` os substituía por `\uFFFD` (`?`).

### Solução
Forçar UTF-8 no processo filho via variável de ambiente `PYTHONUTF8=1`, que instrui o Python a usar UTF-8 para toda I/O independentemente das configurações do sistema:

```python
env = {**os.environ, "PYTHONUTF8": "1"}
result = subprocess.run(cmd, capture_output=True, env=env)
stdout = result.stdout.decode("utf-8", errors="replace")
```

**Limpeza do banco:** faixas já salvas com títulos corrompidos podem ser removidas com:
```bash
python ingest/mongo_crud.py purge-broken --dry-run  # visualizar
python ingest/mongo_crud.py purge-broken             # remover
```

---

## [005] dataset/ criado fora da pasta do projeto

**Data:** Etapa 2 — DSP
**Arquivos:** `ingest/ingest.py`, `dsp/extract_features.py`

### Sintoma
O `extract_features.py` reportava "Arquivo não encontrado" para todas as 1102 faixas, mesmo após a extração ter ocorrido sem erros na Etapa 1.

### Causa
`DATASET_ROOT = Path("dataset")` é um caminho relativo ao CWD no momento da execução. O `ingest.py` foi rodado a partir de `GIT/` (diretório pai do projeto), então os arquivos foram salvos em `GIT/dataset/` em vez de `GIT/music-classifier/dataset/`. O MongoDB armazenou os `file_path` como `dataset\<label>\<titulo>.wav` (relativo), e o `extract_features.py` os resolvia relativamente à raiz do projeto (`music-classifier/`), não encontrando nada.

### Solução

**`ingest/ingest.py`** — usar caminho absoluto relativo ao próprio script:
```python
# Antes
DATASET_ROOT = Path("dataset")

# Depois
DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"
```

**`dsp/extract_features.py`** — tentar dois candidatos ao resolver caminhos relativos:
```python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# No _process_track():
if not file_path.is_absolute():
    candidate = _PROJECT_ROOT / file_path
    if not candidate.exists():
        candidate = _PROJECT_ROOT.parent / file_path  # fallback: diretório pai
    file_path = candidate
```

O fallback para `_PROJECT_ROOT.parent` resolve os dados da execução anterior (em `GIT/dataset/`) sem exigir mover os arquivos.

---

## [006] mongodump falhando com AuthenticationFailed

**Data:** Etapa 2 — Backup
**Arquivo:** `ingest/mongo_crud.py` → `cmd_dump()`, `cmd_restore()`

### Sintoma
```
Failed: can't create session: failed to connect to mongodb://admin:admin123@localhost:27017/:
connection() error occurred during connection handshake: auth error: sasl conversation error:
unable to authenticate using mechanism "SCRAM-SHA-1": (AuthenticationFailed) Authentication failed.
```
Mesmo com URI e credenciais corretos (`admin:admin123@localhost:27017`).

### Causa
O usuário `admin` é criado no banco `admin` (banco de autenticação padrão do MongoDB). O `mongodump`/`mongorestore` não infere automaticamente o banco de autenticação a partir da URI quando o usuário está em `admin` mas o `--db` alvo é outro banco.

### Solução
Passar `--authenticationDatabase admin` explicitamente no comando:

```python
cmd = [MONGODUMP_PATH, "--uri", MONGO_URI, "--authenticationDatabase", "admin",
       "--db", DB_NAME, "--out", str(output_dir)]
```

---
