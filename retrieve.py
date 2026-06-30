"""
Phase B — Retrieval with two modes:

  LIST mode  -> "what properties do you have?", "2BHKs in Pune", "homes with a pool"
                Enumerates EVERY matching listing from the structured table.
  SEMANTIC   -> "inspection findings for the Powai flat", "maintenance fee in Bandra"
                Vector search for the most relevant passages.

Picking the right mode is what fixes "it only mentions 4 properties".

Quick test:
    python retrieve.py "show me all 2bhk in pune"
"""

import json
import re
import sqlite3
from pathlib import Path


DB_PATH     = Path(__file__).resolve().parent / "realestate.db"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K       = 20
LIST_CAP    = 30          # max listings to hand the model in list mode


# ---- lazy heavy deps so the module imports with stdlib alone ----
_embedder = None
def embed(text: str) -> str:
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return json.dumps(_embedder.encode(text, normalize_embeddings=True).tolist())


def connect() -> sqlite3.Connection:
    import sqlite_vec
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.row_factory = sqlite3.Row
    return db


# ------------------------------------------------------------------
# Filter + intent extraction
# ------------------------------------------------------------------
AMENITY_KEYWORDS = [
    "pool", "swimming", "gym", "garden", "parking", "lake", "view", "spa",
    "clubhouse", "balcony", "terrace", "jacuzzi", "tennis", "amphitheatre",
    "furnished", "pet", "security", "power backup", "concierge", "co-working",
]

LIST_TRIGGERS = [
    "all ", "list", "show me", "show all", "what propert", "which propert",
    "which home", "which flat", "which house", "how many", "do you have",
    "available", "options", "properties in", "flats in", "homes in", "houses in",
    "everything", "allow pet", "that have", "with a ", "with an ", "having ",
    "what do you have", "whats available", "what's available", "any ",
]


def extract_filters(db, question: str) -> dict:
    q = question.lower()
    f = {"listing_id": None, "max_price": None, "city": None, "bedrooms": None,
         "property_type": None, "status": None, "keywords": []}

    for (city,) in db.execute("SELECT DISTINCT city FROM listings"):
        if city and city.lower() in q:
            f["city"] = city
    for lid, loc in db.execute("SELECT listing_id, locality FROM listings"):
        if loc and loc.lower() in q:
            f["listing_id"] = lid
    for (pt,) in db.execute("SELECT DISTINCT property_type FROM listings"):
        if pt and pt.lower() in q:
            f["property_type"] = pt
    for (st,) in db.execute("SELECT DISTINCT status FROM listings"):
        if st and st.lower() in q:
            f["status"] = st

    m = re.search(r"(\d+)\s*(?:bhk|bed)", q)
    if m:
        f["bedrooms"] = int(m.group(1))

    m = re.search(r"(?:under|below|less than|upto|up to|within)\s*([\d.]+)\s*(crore|cr|lakh|lac|l)?", q)
    if m:
        amount = float(m.group(1))
        unit = (m.group(2) or "").lower()
        if unit in ("crore", "cr"):
            amount *= 10_000_000
        elif unit in ("lakh", "lac", "l"):
            amount *= 100_000
        f["max_price"] = int(amount)

    for kw in AMENITY_KEYWORDS:
        if kw in q:
            f["keywords"].append(kw)

    return f


def is_list_query(question: str, f: dict) -> bool:
    ql = question.lower()
    if any(t in ql for t in LIST_TRIGGERS):
        return True
    if f["keywords"]:                       # a feature search = enumerate matches
        return True
    return False


# ------------------------------------------------------------------
# LIST mode — pull every matching listing from the structured table
# ------------------------------------------------------------------
def list_listings(db, f: dict) -> list[dict]:
    where, params = [], {}
    if f["city"]:
        where.append("city = :city"); params["city"] = f["city"]
    if f["bedrooms"] is not None:
        where.append("bedrooms = :beds"); params["beds"] = f["bedrooms"]
    if f["property_type"]:
        where.append("property_type = :pt"); params["pt"] = f["property_type"]
    if f["status"]:
        where.append("status = :st"); params["st"] = f["status"]
    if f["max_price"] is not None:
        where.append("price <= :mp"); params["mp"] = f["max_price"]
    for i, kw in enumerate(f["keywords"]):
        where.append(f"LOWER(property_card) LIKE :kw{i}"); params[f"kw{i}"] = f"%{kw}%"

    wsql = (" WHERE " + " AND ".join(where)) if where else ""
    rows = db.execute(
        f"""SELECT listing_id, title, locality, city, price, bedrooms,
                   area_sqft, status, property_card
            FROM listings{wsql}
            ORDER BY price
            LIMIT {LIST_CAP}""",
        params,
    ).fetchall()

    out = []
    for r in rows:
        d = dict(r)
        d["doc_type"] = "card"
        d["source"] = f"listing_{d['listing_id']}_card"
        d["content"] = d.pop("property_card")
        d["distance"] = 0.0
        out.append(d)
    return out


# ------------------------------------------------------------------
# SEMANTIC mode — vector search, then filter
# ------------------------------------------------------------------
def semantic_search(db, question: str, f: dict, top_k: int) -> list[dict]:
    qv = embed(question)
    knn = db.execute(
        """SELECT chunk_id, distance FROM chunk_vectors
           WHERE embedding MATCH ? ORDER BY distance LIMIT ?""",
        (qv, top_k),
    ).fetchall()
    if not knn:
        return []
    dist = {r["chunk_id"]: r["distance"] for r in knn}
    ids = list(dist.keys())

    placeholders = ",".join("?" * len(ids))
    rows = db.execute(
        f"""SELECT c.chunk_id, c.listing_id, c.doc_type, c.source, c.content,
                   l.title, l.price, l.city, l.bedrooms
            FROM chunks c
            LEFT JOIN listings l ON l.listing_id = c.listing_id
            WHERE c.chunk_id IN ({placeholders})""",
        ids,
    ).fetchall()

    out = []
    for r in rows:
        r = dict(r)
        g = r["listing_id"] is None
        if f["listing_id"] is not None and not (r["listing_id"] == f["listing_id"] or g):
            continue
        if f["city"] is not None and not (r["city"] == f["city"] or g):
            continue
        if f["bedrooms"] is not None and not (r["bedrooms"] == f["bedrooms"] or g):
            continue
        if f["max_price"] is not None and not (
            (r["price"] is not None and r["price"] <= f["max_price"]) or g):
            continue
        r["distance"] = dist[r["chunk_id"]]
        out.append(r)
    out.sort(key=lambda r: r["distance"])
    return out


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------
def retrieve(db, question: str, top_k: int = TOP_K):
    f = extract_filters(db, question)
    if is_list_query(question, f):
        return list_listings(db, f), f, "list"
    return semantic_search(db, question, f, top_k), f, "semantic"


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "show me all 2bhk in pune"
    db = connect()
    results, filters, mode = retrieve(db, question)
    print(f"Question: {question}")
    print(f"Mode: {mode}")
    print(f"Filters: { {k: v for k, v in filters.items() if v} }")
    print(f"Matches: {len(results)}\n" + "-" * 70)
    for r in results[:LIST_CAP]:
        tag = f"listing {r['listing_id']}" if r.get('listing_id') else "global"
        print(f"({tag}/{r['doc_type']}) {r['content'][:90]}...")
    db.close()
