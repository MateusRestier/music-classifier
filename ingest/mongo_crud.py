"""
mongo_crud.py — Interface de linha de comando para gerenciar a coleção de faixas no MongoDB.

Uso:
    python ingest/mongo_crud.py <comando> [opções]

Comandos disponíveis:
    stats                          Contagem de faixas por label
    list   [--label L] [--limit N] Lista faixas (todas ou filtradas por label)
    get    <url>                   Exibe os detalhes de uma faixa pela URL
    add    --title T --url U --label L [--file-path F]  Adiciona faixa manualmente
    update <url> [--label L] [--title T] [--file-path F]  Atualiza campos de uma faixa
    delete <url> [--delete-file]   Remove faixa do banco (e opcionalmente o .wav do disco)
"""

import argparse
import os
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


def get_collection() -> Collection:
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]


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
    print(f"\n{'Label':<20} {'Faixas':>8}")
    print("-" * 30)
    for r in results:
        label = r["_id"] or "(sem label)"
        count = r["total"]
        total += count
        print(f"{label:<20} {count:>8}")
    print("-" * 30)
    print(f"{'TOTAL':<20} {total:>8}\n")


def cmd_list(args, col: Collection) -> None:
    """Lista faixas, com filtro opcional por label."""
    query = {}
    if args.label:
        query["label"] = args.label

    limit = args.limit or 0
    cursor = col.find(query, {"_id": 0}).limit(limit)
    docs = list(cursor)

    if not docs:
        print("Nenhuma faixa encontrada.")
        return

    print(f"\n{len(docs)} faixa(s) encontrada(s):\n")
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] {doc.get('title', '—')}  |  {doc.get('label', '—')}")
        print(f"     {doc.get('url', '—')}")
        print(f"     {doc.get('file_path', '—')}\n")


def cmd_search(args, col: Collection) -> None:
    """Busca faixas pelo título (correspondência parcial, sem diferenciar maiúsculas)."""
    import re
    pattern = re.compile(re.escape(args.query), re.IGNORECASE)
    query: dict = {"title": {"$regex": pattern.pattern, "$options": "i"}}
    if args.label:
        query["label"] = args.label

    docs = list(col.find(query, {"_id": 0}).limit(args.limit or 0))

    if not docs:
        print(f'Nenhuma faixa encontrada para "{args.query}".')
        return

    print(f'\n{len(docs)} resultado(s) para "{args.query}":\n')
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] {doc.get('title', '—')}  |  {doc.get('label', '—')}")
        print(f"     {doc.get('url', '—')}\n")


def cmd_get(args, col: Collection) -> None:
    """Exibe detalhes completos de uma faixa pela URL."""
    doc = col.find_one({"url": args.url}, {"_id": 0})
    if not doc:
        print(f"Faixa não encontrada: {args.url}")
        sys.exit(1)
    print(f"\n{_fmt_doc(doc)}\n")


def cmd_add(args, col: Collection) -> None:
    """Insere uma faixa manualmente no banco."""
    existing = col.find_one({"url": args.url})
    if existing:
        print(f"Faixa já existe no banco: {args.url}")
        sys.exit(1)

    doc = {
        "title": args.title,
        "url": args.url,
        "label": args.label,
        "file_path": args.file_path or "",
        "downloaded_at": datetime.now(timezone.utc),
    }
    col.insert_one(doc)
    print(f"\nFaixa adicionada:\n{_fmt_doc(doc)}\n")


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
        print("Nenhum campo para atualizar. Use --label, --title ou --file-path.")
        sys.exit(1)

    result = col.update_one({"url": args.url}, {"$set": updates})
    if result.matched_count == 0:
        print(f"Faixa não encontrada: {args.url}")
        sys.exit(1)

    doc = col.find_one({"url": args.url}, {"_id": 0})
    print(f"\nFaixa atualizada:\n{_fmt_doc(doc)}\n")


def cmd_delete(args, col: Collection) -> None:
    """Remove uma faixa do banco e, opcionalmente, o arquivo .wav do disco."""
    doc = col.find_one({"url": args.url}, {"_id": 0})
    if not doc:
        print(f"Faixa não encontrada: {args.url}")
        sys.exit(1)

    # Confirmação interativa
    print(f"\nFaixa a remover:\n{_fmt_doc(doc)}\n")
    confirm = input("Confirmar remoção? [s/N] ").strip().lower()
    if confirm != "s":
        print("Operação cancelada.")
        return

    col.delete_one({"url": args.url})
    print("Faixa removida do banco.")

    if args.delete_file:
        file_path = doc.get("file_path", "")
        if file_path:
            p = Path(file_path)
            if p.exists():
                p.unlink()
                print(f"Arquivo removido do disco: {p}")
            else:
                print(f"Arquivo não encontrado no disco: {p}")
        else:
            print("Nenhum file_path registrado para esta faixa.")


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
}

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    col = get_collection()
    COMMANDS[args.command](args, col)
