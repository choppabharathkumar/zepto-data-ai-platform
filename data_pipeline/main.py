"""
main.py

Entry point for the data_pipeline module. Running this file end to end:

    1. Scrapes books.toscrape.com (>= 60 books across >= 3 categories)
    2. Saves the raw scraped records to data/raw/books_raw.csv
    3. Loads the results into a pandas DataFrame and validates the dtypes
    4. Saves the cleaned DataFrame to data/processed/books_clean.csv
    5. Writes everything into a fresh SQLite database (categories + books)
    6. Runs the required SQL queries, prints their output, and saves:
         - each query string as  sql/<query_name>.sql
         - each result table as  outputs/query_outputs/<query_name>.txt
    7. Proves pd.read_sql() and pd.merge() agree on the JOIN query

Usage (from the repo root):
    python data_pipeline/main.py

Or from inside the data_pipeline/ directory:
    python main.py
"""

import sys

import pandas as pd

from constants import (
    DB_PATH,
    MIN_BOOKS,
    MIN_CATEGORIES,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    SQL_DIR,
)
from database_manager import get_connection, load_dataframe_to_db
from scraper import scrape_books
from sql_queries import compare_sql_and_pandas_join, run_all_queries


def ensure_directories() -> None:
    """Create all output directories if they don't exist yet."""
    for directory in (RAW_DIR, PROCESSED_DIR, SQL_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_raw_csv(books: list[dict]) -> None:
    """Persist the raw scraped records before any cleaning."""
    raw_path = RAW_DIR / "books_raw.csv"
    pd.DataFrame(books).to_csv(raw_path, index=False)
    print(f"[main] Raw data saved   -> {raw_path}")


def build_dataframe(books: list[dict]) -> pd.DataFrame:
    """Convert the scraped list of dicts into a cleaned, correctly-typed DataFrame."""
    df = pd.DataFrame(books)

    df["title"] = df["title"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df["price_gbp"] = df["price_gbp"].astype(float)
    df["price_inr"] = df["price_inr"].astype(float)
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    df["stock_count"] = df["stock_count"].astype(int)

    # Drop any accidental exact duplicates (same title + category)
    df = df.drop_duplicates(subset=["title", "category"]).reset_index(drop=True)
    return df


def save_processed_csv(df: pd.DataFrame) -> None:
    """Persist the cleaned, typed DataFrame after all transformations."""
    processed_path = PROCESSED_DIR / "books_clean.csv"
    df.to_csv(processed_path, index=False)
    print(f"[main] Processed data saved -> {processed_path}")


def main() -> None:
    ensure_directories()

    # ------------------------------------------------------------------
    # Stage 1: Scrape
    # ------------------------------------------------------------------
    print("[main] Starting scrape...")
    books = scrape_books()

    if len(books) < MIN_BOOKS:
        print(
            f"[main] WARNING: only scraped {len(books)} books, "
            f"expected at least {MIN_BOOKS}. Check your internet connection "
            f"or the site's structure."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 2: Save raw CSV
    # ------------------------------------------------------------------
    save_raw_csv(books)

    # ------------------------------------------------------------------
    # Stage 3: Clean & validate
    # ------------------------------------------------------------------
    books_df = build_dataframe(books)
    categories_found = books_df["category"].nunique()
    print(f"[main] Scraped {len(books_df)} books across {categories_found} categories.")

    if categories_found < MIN_CATEGORIES:
        print(f"[main] WARNING: only {categories_found} categories, expected at least {MIN_CATEGORIES}.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 4: Save processed CSV
    # ------------------------------------------------------------------
    save_processed_csv(books_df)

    # ------------------------------------------------------------------
    # Stage 5: Load into SQLite
    # ------------------------------------------------------------------
    print(f"[main] Loading into SQLite database at '{DB_PATH}'...")
    conn = get_connection(DB_PATH)
    load_dataframe_to_db(conn, books_df)

    # ------------------------------------------------------------------
    # Stage 6: Run SQL queries + save .sql files + save .txt outputs
    # ------------------------------------------------------------------
    print("[main] Running required SQL queries...")
    run_all_queries(conn)

    # ------------------------------------------------------------------
    # Stage 7: Verify pd.read_sql() vs pd.merge() equivalence
    # ------------------------------------------------------------------
    print("\n[main] Verifying pd.read_sql() vs pd.merge() equivalence for the JOIN query...")
    compare_sql_and_pandas_join(conn)

    conn.close()
    print("\n[main] Pipeline complete.")
    print(f"  Raw CSV       -> {RAW_DIR / 'books_raw.csv'}")
    print(f"  Processed CSV -> {PROCESSED_DIR / 'books_clean.csv'}")
    print(f"  Database      -> {DB_PATH}")
    print(f"  SQL files     -> {SQL_DIR}/")
    print(f"  Query outputs -> {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
