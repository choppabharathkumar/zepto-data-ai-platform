"""
main.py

Entry point for the data_pipeline module. Running this file end to end:

    1. Scrapes books.toscrape.com (>= 60 books across >= 3 categories)
    2. Loads the results into a pandas DataFrame and validates the dtypes
    3. Writes everything into a fresh SQLite database (categories + books)
    4. Runs the required SQL queries and prints their output
    5. Proves pd.read_sql() and pd.merge() agree on the JOIN query

Usage:
    python main.py
"""

import sys

import pandas as pd

from constants import DB_PATH, MIN_BOOKS, MIN_CATEGORIES
from database_manager import get_connection, load_dataframe_to_db
from scraper import scrape_books
from sql_queries import compare_sql_and_pandas_join, run_all_queries


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


def main() -> None:
    print("[main] Starting scrape...")
    books = scrape_books()

    if len(books) < MIN_BOOKS:
        print(
            f"[main] WARNING: only scraped {len(books)} books, "
            f"expected at least {MIN_BOOKS}. Check your internet connection "
            f"or the site's structure."
        )
        sys.exit(1)

    books_df = build_dataframe(books)
    categories_found = books_df["category"].nunique()
    print(f"[main] Scraped {len(books_df)} books across {categories_found} categories.")

    if categories_found < MIN_CATEGORIES:
        print(f"[main] WARNING: only {categories_found} categories, expected at least {MIN_CATEGORIES}.")
        sys.exit(1)

    print(f"[main] Loading into SQLite database at '{DB_PATH}'...")
    conn = get_connection(DB_PATH)
    load_dataframe_to_db(conn, books_df)

    print("[main] Running required SQL queries...")
    run_all_queries(conn)

    print("\n[main] Verifying pd.read_sql() vs pd.merge() equivalence for the JOIN query...")
    compare_sql_and_pandas_join(conn)

    conn.close()
    print("\n[main] Pipeline complete.")


if __name__ == "__main__":
    main()
