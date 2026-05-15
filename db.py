import sqlite3
from config import DB_PATH


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS filings (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker       TEXT    NOT NULL,
                cik          TEXT    NOT NULL,
                accession    TEXT    NOT NULL UNIQUE,
                filing_date  TEXT    NOT NULL,
                form_type    TEXT    NOT NULL,
                title        TEXT,
                raw_text     TEXT,
                created_at   TEXT    DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS articles (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker       TEXT    NOT NULL,
                published_at TEXT    NOT NULL,
                title        TEXT,
                description  TEXT,
                content      TEXT,
                url          TEXT    NOT NULL,
                source_name  TEXT,
                relevant     INTEGER DEFAULT 1,
                created_at   TEXT    DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, url)
            );

            CREATE TABLE IF NOT EXISTS prices (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                date            TEXT    NOT NULL,
                open            REAL,
                high            REAL,
                low             REAL,
                close           REAL,
                volume          INTEGER,
                next_day_return REAL,
                created_at      TEXT    DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            );

            CREATE TABLE IF NOT EXISTS sentiment_scores (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                event_date      TEXT    NOT NULL,
                source          TEXT    NOT NULL,
                source_id       INTEGER,
                positive        REAL,
                negative        REAL,
                neutral         REAL,
                composite_score REAL,
                created_at      TEXT    DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_articles_relevant
                ON articles(ticker, relevant);
            CREATE INDEX IF NOT EXISTS idx_filings_ticker_date
                ON filings(ticker, filing_date);
            CREATE INDEX IF NOT EXISTS idx_articles_ticker_date
                ON articles(ticker, published_at);
            CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
                ON prices(ticker, date);
            CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date
                ON sentiment_scores(ticker, event_date);
        """)


def migrate() -> None:
    """Safe migrations for columns added after initial schema creation."""
    with get_conn() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(articles)").fetchall()}
        if "relevant" not in cols:
            conn.execute("ALTER TABLE articles ADD COLUMN relevant INTEGER DEFAULT 1")
            print("Migration: added articles.relevant column")


if __name__ == "__main__":
    init_db()
    migrate()
    print("Database initialized at", DB_PATH)
