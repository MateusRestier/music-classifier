"""
mongo_crud.py — Interface de linha de comando para gerenciar a coleção de faixas no MongoDB.

Uso:
    python ingest/mongo_crud.py <comando> [opções]

Comandos disponíveis:
    stats                          Contagem de faixas por label
    list   [--label L] [--limit N] Lista faixas (todas ou filtradas por label)
    search <query> [--label L] [--limit N]  Busca faixas por trecho do título
    get    <url>                   Exibe os detalhes de uma faixa pela URL
    add    --title T --url U --label L [--file-path F]  Adiciona faixa manualmente
    update <url> [--label L] [--title T] [--file-path F]  Atualiza campos de uma faixa
    delete <url> [--delete-file]   Remove faixa do banco (e opcionalmente o .wav do disco)
    purge-broken [--dry-run]       Remove faixas com títulos corrompidos (caracteres ?)
    dump   [--output DIR]          Exporta o banco para um diretório via mongodump
    restore <dump_dir> [--drop]    Restaura o banco a partir de um dump via mongorestore
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.collection import Collection

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:admin123@localhost:27017/")
DB_NAME = "music_classifier"
COLLECTION_NAME = "tracks"

_MONGO_TOOLS_BIN = r"C:\Program Files\MongoDB\Tools\100\bin"
MONGODUMP_PATH    = os.getenv("MONGODUMP_PATH",    rf"{_MONGO_TOOLS_BIN}\mongodump.exe")
MONGORESTORE_PATH = os.getenv("MONGORESTORE_PATH", rf"{_MONGO_TOOLS_BIN}\mongorestore.exe")


def get_collection() -> Collection:
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]


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


logger = _setup_logging("mongo_crud")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_doc(doc: dict) -> str:
    lines = [
        f"  title      : {doc.get('title', '—')}",
        f"  url        : {doc.get('url', '—')}",
        f"  label      : {doc.get('label', '—')}",
        f"  file_path  : {doc.get('file_path', '—')}",
        f"  downloaded : {doc.get('downloaded_at', '—')}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------

def cmd_stats(args, col: Collection) -> None:
    """Exibe a contagem de faixas por label."""
    pipeline = [{"$group": {"_id": "$label", "total": {"$sum": 1}}}]
    results = sorted(col.aggregate(pipeline), key=lambda x: x["_id"] or "")
    total = 0
    logger.info("\n%-20s %8s", "Label", "Faixas")
    logger.info("-" * 30)
    for r in results:
        label = r["_id"] or "(sem label)"
        count = r["total"]
        total += count
        logger.info("%-20s %8d", label, count)
    logger.info("-" * 30)
    logger.info("%-20s %8d\n", "TOTAL", total)


def cmd_list(args, col: Collection) -> None:
    """Lista faixas, com filtro opcional por label."""
    query = {}
    if args.label:
        query["label"] = args.label

    limit = args.limit or 0
    cursor = col.find(query, {"_id": 0}).limit(limit)
    docs = list(cursor)

    if not docs:
        logger.info("Nenhuma faixa encontrada.")
        return

    logger.info("\n%d faixa(s) encontrada(s):\n", len(docs))
    for i, doc in enumerate(docs, 1):
        logger.info("[%d] %s  |  %s", i, doc.get("title", "—"), doc.get("label", "—"))
        logger.info("     %s", doc.get("url", "—"))
        logger.info("     %s\n", doc.get("file_path", "—"))


def cmd_search(args, col: Collection) -> None:
    """Busca faixas pelo título (correspondência parcial, sem diferenciar maiúsculas)."""
    import re
    pattern = re.compile(re.escape(args.query), re.IGNORECASE)
    query: dict = {"title": {"$regex": pattern.pattern, "$options": "i"}}
    if args.label:
        query["label"] = args.label

    docs = list(col.find(query, {"_id": 0}).limit(args.limit or 0))

    if not docs:
        logger.info('Nenhuma faixa encontrada para "%s".', args.query)
        return

    logger.info('\n%d resultado(s) para "%s":\n', len(docs), args.query)
    for i, doc in enumerate(docs, 1):
        logger.info("[%d] %s  |  %s", i, doc.get("title", "—"), doc.get("label", "—"))
        logger.info("     %s\n", doc.get("url", "—"))


def cmd_get(args, col: Collection) -> None:
    """Exibe detalhes completos de uma faixa pela URL."""
    doc = col.find_one({"url": args.url}, {"_id": 0})
    if not doc:
        logger.error("Faixa não encontrada: %s", args.url)
        sys.exit(1)
    logger.info("\n%s\n", _fmt_doc(doc))


def cmd_add(args, col: Collection) -> None:
    """Insere uma faixa manualmente no banco."""
    existing = col.find_one({"url": args.url})
    if existing:
        logger.warning("Faixa já existe no banco: %s", args.url)
        sys.exit(1)

    doc = {
        "title": args.title,
        "url": args.url,
        "label": args.label,
        "file_path": args.file_path or "",
        "downloaded_at": datetime.now(timezone.utc),
    }
    col.insert_one(doc)
    logger.info("\nFaixa adicionada:\n%s\n", _fmt_doc(doc))


def cmd_update(args, col: Collection) -> None:
    """Atualiza campos de uma faixa existente."""
    updates = {}
    if args.label:
        updates["label"] = args.label
    if args.title:
        updates["title"] = args.title
    if args.file_path:
        updates["file_path"] = args.file_path

    if not updates:
        logger.warning("Nenhum campo para atualizar. Use --label, --title ou --file-path.")
        sys.exit(1)

    result = col.update_one({"url": args.url}, {"$set": updates})
    if result.matched_count == 0:
        logger.error("Faixa não encontrada: %s", args.url)
        sys.exit(1)

    doc = col.find_one({"url": args.url}, {"_id": 0})
    logger.info("\nFaixa atualizada:\n%s\n", _fmt_doc(doc))


def cmd_delete(args, col: Collection) -> None:
    """Remove uma faixa do banco e, opcionalmente, o arquivo .wav do disco."""
    doc = col.find_one({"url": args.url}, {"_id": 0})
    if not doc:
        logger.error("Faixa não encontrada: %s", args.url)
        sys.exit(1)

    # Confirmação interativa
    print(f"\nFaixa a remover:\n{_fmt_doc(doc)}\n")
    confirm = input("Confirmar remoção? [s/N] ").strip().lower()
    if confirm != "s":
        logger.info("Operação cancelada.")
        return

    col.delete_one({"url": args.url})
    logger.info("Faixa removida do banco.")

    if args.delete_file:
        file_path = doc.get("file_path", "")
        if file_path:
            p = Path(file_path)
            if p.exists():
                p.unlink()
                logger.info("Arquivo removido do disco: %s", p)
            else:
                logger.warning("Arquivo não encontrado no disco: %s", p)
        else:
            logger.warning("Nenhum file_path registrado para esta faixa.")


def cmd_purge_broken(args, col: Collection) -> None:
    """Remove faixas com títulos corrompidos (contêm o caractere de substituição Unicode U+FFFD)."""
    query = {"title": {"$regex": "\ufffd"}}
    docs = list(col.find(query, {"_id": 0}))

    if not docs:
        logger.info("Nenhuma faixa com título corrompido encontrada.")
        return

    logger.info("\n%d faixa(s) com título corrompido:\n", len(docs))
    for i, doc in enumerate(docs, 1):
        logger.info("[%d] %s  |  %s", i, doc.get("title", "—"), doc.get("label", "—"))
        logger.info("     %s\n", doc.get("url", "—"))

    if args.dry_run:
        logger.info("[DRY-RUN] Nenhuma faixa removida.")
        return

    confirm = input(f"Remover as {len(docs)} faixa(s) listadas acima? [s/N] ").strip().lower()
    if confirm != "s":
        logger.info("Operação cancelada.")
        return

    result = col.delete_many(query)
    logger.info("%d faixa(s) removida(s) do banco.", result.deleted_count)


_BACKUPS_DIR = Path(__file__).resolve().parent.parent / "backups"


def cmd_dump(args, col: Collection) -> None:
    """Exporta o banco para um diretório via mongodump."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) if args.output else _BACKUPS_DIR / f"dump_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [MONGODUMP_PATH, "--uri", MONGO_URI, "--authenticationDatabase", "admin",
           "--db", DB_NAME, "--out", str(output_dir)]
    logger.info("Executando: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True)
    _log_subprocess_output(result)

    if result.returncode != 0:
        logger.error("mongodump falhou (código %d).", result.returncode)
        sys.exit(result.returncode)

    logger.info("Backup salvo em: %s", output_dir / DB_NAME)


def cmd_restore(args, col: Collection) -> None:
    """Restaura o banco a partir de um dump via mongorestore."""
    dump_dir = Path(args.dump_dir)

    # mongodump cria subdiretório com o nome do banco; aceita ambos os formatos
    db_dir = dump_dir / DB_NAME if (dump_dir / DB_NAME).exists() else dump_dir
    if not db_dir.exists():
        logger.error("Diretório de dump não encontrado: %s", dump_dir)
        sys.exit(1)

    cmd = [MONGORESTORE_PATH, "--uri", MONGO_URI, "--authenticationDatabase", "admin",
           "--db", DB_NAME]
    if args.drop:
        cmd.append("--drop")
    cmd.append(str(db_dir))

    logger.info("Executando: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True)
    _log_subprocess_output(result)

    if result.returncode != 0:
        logger.error("mongorestore falhou (código %d).", result.returncode)
        sys.exit(result.returncode)

    logger.info("Banco restaurado com sucesso.")


def _log_subprocess_output(result: subprocess.CompletedProcess) -> None:
    """Loga stdout e stderr de um subprocess (mongodump/mongorestore escrevem em stderr)."""
    for stream in (result.stdout, result.stderr):
        if stream:
            text = stream.decode("utf-8", errors="replace").strip()
            if text:
                logger.info(text)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CRUD de faixas no MongoDB do music-classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # stats
    sub.add_parser("stats", help="Contagem de faixas por label")

    # list
    p_list = sub.add_parser("list", help="Lista faixas")
    p_list.add_argument("--label", "-l", help="Filtrar por label (ex: metalcore)")
    p_list.add_argument("--limit", "-n", type=int, default=50, help="Máximo de resultados (padrão: 50)")

    # search
    p_search = sub.add_parser("search", help="Busca faixas por trecho do título")
    p_search.add_argument("query", help='Trecho a buscar (ex: "the end")')
    p_search.add_argument("--label", "-l", help="Restringir busca a um label")
    p_search.add_argument("--limit", "-n", type=int, default=20, help="Máximo de resultados (padrão: 20)")

    # get
    p_get = sub.add_parser("get", help="Detalhes de uma faixa pela URL")
    p_get.add_argument("url", help="URL do YouTube da faixa")

    # add
    p_add = sub.add_parser("add", help="Adiciona uma faixa manualmente")
    p_add.add_argument("--title", "-t", required=True, help="Título da faixa")
    p_add.add_argument("--url", "-u", required=True, help="URL do YouTube")
    p_add.add_argument("--label", "-l", required=True, help="Label/categoria (ex: metalcore)")
    p_add.add_argument("--file-path", "-f", help="Caminho local do arquivo .wav (opcional)")

    # update
    p_upd = sub.add_parser("update", help="Atualiza campos de uma faixa")
    p_upd.add_argument("url", help="URL do YouTube da faixa a atualizar")
    p_upd.add_argument("--label", "-l", help="Novo label")
    p_upd.add_argument("--title", "-t", help="Novo título")
    p_upd.add_argument("--file-path", "-f", help="Novo caminho do arquivo")

    # delete
    p_del = sub.add_parser("delete", help="Remove uma faixa do banco")
    p_del.add_argument("url", help="URL do YouTube da faixa a remover")
    p_del.add_argument(
        "--delete-file", action="store_true",
        help="Apaga também o arquivo .wav do disco"
    )

    # purge-broken
    p_purge = sub.add_parser("purge-broken", help="Remove faixas com títulos corrompidos")
    p_purge.add_argument(
        "--dry-run", action="store_true",
        help="Apenas lista as faixas, sem remover"
    )

    # dump
    p_dump = sub.add_parser("dump", help="Exporta o banco via mongodump")
    p_dump.add_argument(
        "--output", "-o",
        help=f"Diretório de saída (padrão: backups/dump_YYYYMMDD_HHMMSS)"
    )

    # restore
    p_restore = sub.add_parser("restore", help="Restaura o banco via mongorestore")
    p_restore.add_argument("dump_dir", help="Diretório do dump (ex: backups/dump_20260311_150000)")
    p_restore.add_argument(
        "--drop", action="store_true",
        help="Apaga a coleção existente antes de restaurar (evita duplicatas)"
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {
    "stats": cmd_stats,
    "list": cmd_list,
    "search": cmd_search,
    "get": cmd_get,
    "add": cmd_add,
    "update": cmd_update,
    "delete": cmd_delete,
    "purge-broken": cmd_purge_broken,
    "dump": cmd_dump,
    "restore": cmd_restore,
}

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    col = get_collection()
    COMMANDS[args.command](args, col)
