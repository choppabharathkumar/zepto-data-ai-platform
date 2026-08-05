"""
sql_queries.py

Every required SQL query lives here as a named string, plus the helper
functions that run them and that show pd.read_sql / pd.merge produce
equivalent results.

Required clause coverage (capstone brief):
    SELECT / WHERE  -> books_in_stock_high_rating
    ORDER BY / LIMIT -> ten_most_expensive_books
    DISTINCT        -> distinct_ratings
    IN              -> books_in_selected_categories
    BETWEEN         -> midrange_priced_books
    JOIN            -> top_rated_books_by_category  (also uses ORDER BY + LIMIT)
"""

import sqlite3

import pandas as pd

from constants import OUTPUTS_DIR, SQL_DIR

# --------------------------------------------------------------------------
# Named queries
# --------------------------------------------------------------------------
QUERIES: dict[str, str] = {
    "books_in_stock_high_rating": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE in_stock = 1 AND rating >= 4;
    """,
    "ten_most_expensive_books": """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,
    "distinct_ratings": """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
    """,
    "books_in_selected_categories": """
        SELECT b.title, c.category_name, b.price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE c.category_name IN ('Travel', 'Mystery', 'Historical Fiction');
    """,
    "midrange_priced_books": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 10 AND 30
        ORDER BY price_gbp;
    """,
    "top_rated_books_by_category": """
        SELECT c.category_name, b.title, b.rating, b.price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.price_gbp ASC
        LIMIT 10;
    """,
}


def run_query(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    """Execute one named query and return its result as a DataFrame."""
    return pd.read_sql(QUERIES[name], conn)


def save_queries_to_sql_files() -> None:
    """
    Write every named query to its own .sql file inside SQL_DIR.
    This makes each query independently readable on GitHub without
    having to open Python source.
    """
    SQL_DIR.mkdir(parents=True, exist_ok=True)
    for name, sql in QUERIES.items():
        sql_path = SQL_DIR / f"{name}.sql"
        sql_path.write_text(sql.strip(), encoding="utf-8")
    print(f"[sql_queries] SQL files saved -> {SQL_DIR}/")


def save_query_outputs_to_txt(results: dict[str, pd.DataFrame]) -> None:
    """
    Write the result DataFrame of every query to a plain-text file
    inside OUTPUTS_DIR so the outputs are visible on GitHub without
    running the pipeline.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        out_path = OUTPUTS_DIR / f"{name}.txt"
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(f"-- {name} --\n\n")
            fh.write(df.to_string(index=False))
            fh.write("\n")
    print(f"[sql_queries] Query outputs saved -> {OUTPUTS_DIR}/")


def run_all_queries(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    """
    Execute every query in QUERIES, print its SQL and its output, and
    return {query_name: result_dataframe} so callers can reuse the results.

    Also persists:
      - each query string as  sql/<query_name>.sql
      - each result table as  outputs/query_outputs/<query_name>.txt
    """
    results = {}
    for name, sql in QUERIES.items():
        print(f"\n--- {name} ---")
        print(sql.strip())
        df = run_query(conn, name)
        print(df.to_string(index=False))
        results[name] = df

    # Persist artifacts
    save_queries_to_sql_files()
    save_query_outputs_to_txt(results)

    return results


def merge_top_rated_books_by_category(books_df: pd.DataFrame, categories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the "top_rated_books_by_category" SQL query using only
    pandas (pd.merge on the in-memory DataFrames, no SQL involved). Used
    to prove that the SQL JOIN and the pandas merge agree with each other.
    """
    merged = books_df.merge(categories_df, on="category_id", how="inner")
    merged = merged.sort_values(by=["rating", "price_gbp"], ascending=[False, True])
    result = merged[["category_name", "title", "rating", "price_gbp"]].head(10)
    return result.reset_index(drop=True)


def compare_sql_and_pandas_join(conn: sqlite3.Connection) -> bool:
    """
    Run the JOIN query two independent ways and confirm they match:
      1. pd.read_sql() against the SQL JOIN query.
      2. pd.merge() against the raw books/categories tables read into
         DataFrames (no SQL join at all).

    Returns True if both approaches produce the same rows in the same
    order, and prints both DataFrames side by side either way.
    """
    sql_result = run_query(conn, "top_rated_books_by_category").reset_index(drop=True)

    books_df = pd.read_sql("SELECT * FROM books;", conn)
    categories_df = pd.read_sql("SELECT * FROM categories;", conn)
    pandas_result = merge_top_rated_books_by_category(books_df, categories_df)

    print("\n--- pd.read_sql() result (SQL JOIN) ---")
    print(sql_result.to_string(index=False))

    print("\n--- pd.merge() result (in-memory, no SQL) ---")
    print(pandas_result.to_string(index=False))

    are_equal = sql_result.equals(pandas_result)
    print(f"\nResults match: {are_equal}")
    return are_equal
