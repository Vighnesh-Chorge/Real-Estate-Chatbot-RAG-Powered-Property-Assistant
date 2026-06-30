"""
Phase A — Ingestion. Reads property data from many file formats, categorizes
each property into structured features, and stores everything in realestate.db.

Supported formats (drop files into the project's  data/  folder):
    .csv  .tsv  .json  .jsonl        -> no extra deps
    .xlsx .xls  .parquet             -> needs:  pip install pandas openpyxl

If no data files are found, it falls back to built-in sample listings so the
pipeline always runs.

Run:
    pip install sentence-transformers sqlite-vec PyMuPDF
    python ingest/ingest.py
"""

import csv
import json
import math
import os
import re
from pathlib import Path


# ------------------------------------------------------------------
# Paths & config
# ------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent.parent   # project root
DB_PATH       = ROOT / "realestate.db"
SCHEMA_SQL    = ROOT / "Core" / "schema.sql"
DATA_DIR      = ROOT / "data"                             # put your data files here
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"                  # 384-dim, CPU-friendly
EMBED_DIM     = 384
CHUNK_WORDS   = 320
CHUNK_OVERLAP = 40

SUPPORTED = {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls", ".parquet"}


# ==================================================================
# 1) READERS — turn any supported file into a list of raw dict rows
# ==================================================================
def load_records(path: Path) -> list[dict]:
    ext = path.suffix.lower()
    if ext == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    if ext == ".tsv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f, delimiter="\t"))
    if ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):                    # {"listings": [...]} or a single record
            data = data.get("listings") or data.get("records") or [data]
        return list(data)
    if ext in (".xlsx", ".xls", ".parquet"):
        import pandas as pd                           # lazy: only needed for these
        df = pd.read_parquet(path) if ext == ".parquet" else pd.read_excel(path)
        df = df.where(df.notna(), None)               # NaN -> None
        return df.to_dict(orient="records")
    raise ValueError(f"Unsupported file type: {path.name}")


# ==================================================================
# 2) NORMALIZE — map varied column names to canonical features
# ==================================================================
# canonical field -> the many header names that should map to it
FIELD_ALIASES = {
    "listing_id":     ["listing_id", "id", "property_id", "ref", "reference"],
    "title":          ["title", "name", "headline", "listing_title"],
    "property_type":  ["property_type", "type", "category", "propertytype"],
    "price":          ["price", "cost", "amount", "asking_price", "value", "rate"],
    "bedrooms":       ["bedrooms", "beds", "bhk", "bedroom", "no_of_bedrooms"],
    "bathrooms":      ["bathrooms", "baths", "bathroom", "washrooms", "toilets"],
    "area_sqft":      ["area_sqft", "sqft", "square_feet", "area", "carpet_area",
                       "builtup_area", "built_up_area", "super_builtup_area", "size"],
    "floor":          ["floor", "floor_no", "floor_number", "level"],
    "facing":         ["facing", "direction", "facing_direction"],
    "parking":        ["parking", "parking_spots", "car_parking", "parking_count"],
    "furnishing":     ["furnishing", "furnished", "furnishing_status"],
    "year_built":     ["year_built", "built_year", "construction_year", "yearbuilt"],
    "maintenance_fee":["maintenance_fee", "maintenance", "monthly_maintenance",
                       "hoa", "hoa_fee", "society_charges"],
    "locality":       ["locality", "area_name", "neighborhood", "neighbourhood",
                       "sublocality", "region"],
    "city":           ["city", "town"],
    "status":         ["status", "availability", "listing_status"],
    "description":    ["description", "details", "about", "overview", "remarks"],
}
# reverse lookup: normalized header -> canonical field
ALIAS_TO_CANON = {a: canon for canon, aliases in FIELD_ALIASES.items() for a in aliases}

INT_FIELDS   = {"bedrooms", "parking", "year_built"}
FLOAT_FIELDS = {"bathrooms"}
MONEY_FIELDS = {"price", "maintenance_fee"}
AREA_FIELDS  = {"area_sqft"}

DOC_HINTS = ("brochure", "inspection", "faq", "document", "pdf", "floorplan", "report")


def norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(k).strip().lower()).strip("_")


def parse_money(v) -> int | None:
    """'4.5 Cr' -> 45000000 ; '18,000' -> 18000 ; '45 lakh' -> 4500000."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return int(v)
    s = str(v).lower().replace(",", "")
    m = re.search(r"([\d.]+)", s)
    if not m:
        return None
    amount = float(m.group(1))
    if "cr" in s:
        amount *= 10_000_000
    elif "lakh" in s or "lac" in s:
        amount *= 100_000
    elif re.search(r"\bk\b", s):
        amount *= 1_000
    return int(amount)


def parse_int(v) -> int | None:
    if v is None:
        return None
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else None


def parse_float(v) -> float | None:
    if v is None:
        return None
    m = re.search(r"([\d.]+)", str(v))
    return float(m.group(1)) if m else None


def normalize_record(raw: dict, fallback_id: int) -> tuple[dict, dict, list]:
    """Return (listing_core, extra_features, documents) for one raw row."""
    listing, features = {}, {}
    documents = list(raw["documents"]) if isinstance(raw.get("documents"), list) else []

    for k, v in raw.items():
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        nk = norm_key(k)
        if nk == "documents":
            continue
        canon = ALIAS_TO_CANON.get(nk)
        if canon:
            listing[canon] = v
        elif any(h in nk for h in DOC_HINTS):
            # a document column: a .pdf path, or inline text
            val = str(v)
            dtype = next((h for h in ("brochure", "inspection", "faq") if h in nk), "document")
            documents.append({"doc_type": dtype, "path": val} if val.lower().endswith(".pdf")
                             else {"doc_type": dtype, "text": val})
        else:
            features[nk] = v                          # any other category

    # type-clean the core fields
    for f in MONEY_FIELDS:
        if f in listing: listing[f] = parse_money(listing[f])
    for f in AREA_FIELDS:
        if f in listing: listing[f] = parse_int(listing[f])
    for f in INT_FIELDS:
        if f in listing: listing[f] = parse_int(listing[f])
    for f in FLOAT_FIELDS:
        if f in listing: listing[f] = parse_float(listing[f])

    # description also becomes a searchable document
    if listing.get("description"):
        documents.append({"doc_type": "description", "text": str(listing["description"])})

    listing["listing_id"] = parse_int(listing.get("listing_id")) or fallback_id
    return listing, features, documents


# ==================================================================
# 3) PROPERTY CARD — natural-language summary (gets embedded)
# ==================================================================
def make_property_card(listing: dict, features: dict) -> str:
    p = []
    if listing.get("title"):
        p.append(f"{listing['title']}.")
    bits = []
    if listing.get("bedrooms") is not None:  bits.append(f"{listing['bedrooms']}-bedroom")
    if listing.get("bathrooms") is not None: bits.append(f"{listing['bathrooms']}-bath")
    if listing.get("property_type"):         bits.append(str(listing["property_type"]).lower())
    if bits:
        p.append("A " + ", ".join(bits) + " property.")
    if listing.get("area_sqft"):
        p.append(f"Area: {listing['area_sqft']} sqft.")
    loc = ", ".join(x for x in [listing.get("locality"), listing.get("city")] if x)
    if loc:
        p.append(f"Located in {loc}.")
    if listing.get("price"):
        p.append(f"Priced at {listing['price']} rupees.")
    for k in ("floor", "facing", "parking", "furnishing", "year_built",
              "maintenance_fee", "status"):
        if listing.get(k) is not None:
            p.append(f"{k.replace('_', ' ').title()}: {listing[k]}.")
    for fname, fval in features.items():
        p.append(f"{fname.replace('_', ' ').title()}: {fval}.")
    return " ".join(p)


# ==================================================================
# 4) Embedding + DB plumbing
# ==================================================================
_embedder = None
def embed(text: str) -> list[float]:
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder.encode(text, normalize_embeddings=True).tolist()


def connect():
    import sqlite3, sqlite_vec
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def init_schema(db):
    db.executescript(Path(SCHEMA_SQL).read_text())
    db.commit()


def get_doc_text(doc: dict) -> str:
    if "text" in doc:
        return doc["text"]
    import fitz
    pdf = fitz.open(doc["path"])
    return "\n".join(page.get_text() for page in pdf)


def chunk_text(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    out, start, step = [], 0, CHUNK_WORDS - CHUNK_OVERLAP
    while start < len(words):
        out.append(" ".join(words[start:start + CHUNK_WORDS]))
        start += step
    return out


def insert_chunk(db, listing_id, doc_type, source, content):
    cur = db.execute(
        "INSERT INTO chunks (listing_id, doc_type, source, content) VALUES (?, ?, ?, ?)",
        (listing_id, doc_type, source, content))
    db.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (cur.lastrowid, json.dumps(embed(content))))


# ==================================================================
# 5) Ingest one normalized listing
# ==================================================================
LISTING_COLUMNS = ["listing_id", "title", "property_type", "price", "bedrooms",
                   "bathrooms", "area_sqft", "floor", "facing", "parking",
                   "furnishing", "year_built", "maintenance_fee", "locality",
                   "city", "status", "description"]


def ingest_listing(db, listing: dict, features: dict, documents: list):
    card = make_property_card(listing, features)
    row = {c: listing.get(c) for c in LISTING_COLUMNS}
    row["property_card"] = card
    cols = ", ".join(row.keys())
    ph = ", ".join(f":{k}" for k in row.keys())
    db.execute(f"INSERT OR REPLACE INTO listings ({cols}) VALUES ({ph})", row)

    for name, value in features.items():            # store every extra feature/category
        db.execute(
            "INSERT INTO listing_features (listing_id, feature_name, feature_value) VALUES (?, ?, ?)",
            (listing["listing_id"], name, str(value)))

    insert_chunk(db, listing["listing_id"], "card",
                 f"listing_{listing['listing_id']}_card", card)
    for d in documents:
        src = d.get("source") or f"listing_{listing['listing_id']}_{d['doc_type']}"
        for piece in chunk_text(get_doc_text(d)):
            insert_chunk(db, listing["listing_id"], d["doc_type"], src, piece)


# ==================================================================
# 6) Gather input (from data/ files, or built-in sample)
# ==================================================================
def gather_records() -> list[dict]:
    records = []
    if DATA_DIR.exists():
        for p in sorted(DATA_DIR.iterdir()):
            if p.suffix.lower() in SUPPORTED:
                found = load_records(p)
                print(f"  read {len(found):>4} records from {p.name}")
                records.extend(found)
    if not records:
        print("  no data files in ./data — using built-in sample listings")
        records = sample_listings()
    return records


def sample_listings() -> list[dict]:
    """Fallback demo data so the pipeline runs with zero input files."""
    return [
        {"listing_id": 482, "title": "Sunlit 3BHK near the metro", "property_type": "apartment",
         "price": "4.5 Cr", "bedrooms": 3, "bathrooms": 2, "area_sqft": 1400, "floor": "11th",
         "facing": "west", "parking": 2, "furnishing": "semi-furnished", "year_built": 2013,
         "maintenance_fee": 18000, "locality": "Bandra West", "city": "Mumbai",
         "status": "available", "amenities": "rooftop gym, pool, play area", "pet_policy": "allowed",
         "documents": [
             {"doc_type": "inspection", "text": "Inspection March 2025: structure sound. "
              "Minor seepage in the guest bathroom ceiling was repaired in 2024."}]},
        {"listing_id": 603, "title": "Spacious 4BHK with lake view", "property_type": "apartment",
         "price": "6.2 Cr", "bedrooms": 4, "bathrooms": 4, "area_sqft": 2200, "floor": "18th",
         "facing": "north", "parking": 3, "furnishing": "furnished", "year_built": 2017,
         "maintenance_fee": 32000, "locality": "Powai", "city": "Mumbai", "status": "available",
         "amenities": "spa, indoor games, 25m pool", "view": "Powai Lake", "pet_policy": "allowed",
         "documents": [
             {"doc_type": "brochure", "text": "Expansive 4BHK on the 18th floor of Lakeside "
              "Towers with uninterrupted views of Powai Lake. Walk to schools and the market."}]},
    ]


# ==================================================================
# 7) Orchestration
# ==================================================================
def main():
    print(f"Gathering data (looking in {DATA_DIR}) ...")
    records = gather_records()

    db = connect()
    init_schema(db)                                 # ensure tables exist
    # Clear rows instead of deleting the file -- avoids Windows/OneDrive file locks.
    for t in ("chunk_vectors", "chunks", "listing_features", "listings"):
        db.execute(f"DELETE FROM {t}")
    db.commit()
    for i, raw in enumerate(records, start=1):
        listing, features, documents = normalize_record(raw, fallback_id=i)
        ingest_listing(db, listing, features, documents)
    db.commit()

    nl = db.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    nf = db.execute("SELECT COUNT(*) FROM listing_features").fetchone()[0]
    nc = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    nv = db.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
    print(f"\nIngested {nl} listings, {nf} extra features, {nc} chunks, {nv} vectors.")
    print("OK: vectors stored." if nv == nc else "WARNING: vector count != chunk count.")
    db.close()


if __name__ == "__main__":
    main()
