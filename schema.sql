-- ============================================================
--  Real-estate RAG — schema (SQLite + sqlite-vec)
--  Expanded: many structured feature columns + a flexible
--  key-value table for any extra features your data contains.
-- ============================================================

-- ------------------------------------------------------------
--  1) Listings — the structured features you FILTER on.
--     Core categories are typed columns; anything unusual in
--     your data goes into listing_features (below) instead.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS listings (
    listing_id      INTEGER PRIMARY KEY,
    title           TEXT,
    property_type   TEXT,      -- apartment | villa | plot | studio ...
    price           INTEGER,   -- in rupees, cleaned to a plain number
    bedrooms        INTEGER,
    bathrooms       REAL,      -- REAL allows 2.5
    area_sqft       INTEGER,
    floor           TEXT,
    facing          TEXT,      -- north | south | east | west ...
    parking         INTEGER,   -- number of spots
    furnishing      TEXT,      -- furnished | semi | unfurnished
    year_built      INTEGER,
    maintenance_fee INTEGER,   -- monthly, in rupees
    locality        TEXT,
    city            TEXT,
    status          TEXT,      -- available | sold | pending
    description     TEXT,      -- free text (also chunked as a document)
    property_card   TEXT       -- generated NL summary (embedded for search)
);

-- ------------------------------------------------------------
--  2) Flexible features — any category your data has that isn't
--     a core column above (amenities, pet_policy, balcony, gym,
--     possession_date, rera_id, etc.). One row per feature.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS listing_features (
    listing_id    INTEGER,
    feature_name  TEXT,
    feature_value TEXT,
    FOREIGN KEY (listing_id) REFERENCES listings(listing_id)
);
CREATE INDEX IF NOT EXISTS idx_features_listing ON listing_features(listing_id);
CREATE INDEX IF NOT EXISTS idx_features_name    ON listing_features(feature_name);

-- ------------------------------------------------------------
--  3) Text chunks — every searchable passage WITH its metadata.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id  INTEGER,                 -- NULL for global docs
    doc_type    TEXT NOT NULL,           -- card | description | brochure | inspection | faq | global
    source      TEXT,
    content     TEXT NOT NULL,
    FOREIGN KEY (listing_id) REFERENCES listings(listing_id)
);
CREATE INDEX IF NOT EXISTS idx_chunks_listing ON chunks(listing_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doctype ON chunks(doc_type);

-- ------------------------------------------------------------
--  4) Vector index (sqlite-vec). 384 dims = bge-small-en.
-- ------------------------------------------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
    chunk_id  INTEGER PRIMARY KEY,
    embedding FLOAT[384]
);
