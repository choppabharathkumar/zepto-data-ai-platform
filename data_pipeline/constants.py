"""
constants.py

Every "magic value" used across the data_pipeline module lives here so that
nothing is hard-coded inside the business-logic files. If the site URL, the
exchange rate, or the database path ever need to change, this is the only
file that should need editing.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Directory layout
# --------------------------------------------------------------------------
# All paths are anchored to *this file's* location so the pipeline always
# writes its outputs inside data_pipeline/ regardless of where you invoke
# `python` from (e.g. repo root vs. inside the subdirectory).
MODULE_DIR: Path = Path(__file__).parent

# Intermediate data artefacts
DATA_DIR: Path = MODULE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# SQL query files (one .sql per named query)
SQL_DIR: Path = MODULE_DIR / "sql"

# Human-readable query result files (one .txt per named query)
OUTPUTS_DIR: Path = MODULE_DIR / "outputs" / "query_outputs"

# SQLite database
DB_PATH: Path = MODULE_DIR / "books.db"

# --------------------------------------------------------------------------
# Source website
# --------------------------------------------------------------------------
BASE_URL: str = "https://books.toscrape.com/"

# --------------------------------------------------------------------------
# Currency conversion
# --------------------------------------------------------------------------
# Fixed, project-defined exchange rate. This is NOT pulled from a live API -
# the capstone brief requires a constant rate with no date reference.
GBP_TO_INR: float = 105.50

# --------------------------------------------------------------------------
# Scraping targets (acceptance criteria from the capstone brief)
# --------------------------------------------------------------------------
MIN_CATEGORIES: int = 3
MIN_BOOKS: int = 60

# --------------------------------------------------------------------------
# HTTP behaviour
# --------------------------------------------------------------------------
REQUEST_TIMEOUT_SECONDS: int = 15
REQUEST_RETRIES: int = 3
RETRY_BACKOFF_SECONDS: float = 1.5
REQUEST_HEADERS: dict = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ZeptoCapstoneBot/1.0; "
        "+for-education-purposes-only)"
    )
}

# Be polite to the (free, static, scraping-practice) target site.
POLITE_DELAY_SECONDS: float = 0.2

# --------------------------------------------------------------------------
# Rating words used on the site map to these integers.
# --------------------------------------------------------------------------
RATING_WORD_TO_INT: dict = {
    "Zero": 0,
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}
