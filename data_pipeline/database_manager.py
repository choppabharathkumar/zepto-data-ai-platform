"""
database_manager.py

Everything related to talking to SQLite lives here: creating the
categories/books schema, and loading a cleaned pandas DataFrame into it.
"""

import sqlite3

import pandas as pd

from constants import DB_PATH

CREATE_CATEGORIES_TABLE = """
CREATE TABLE IF NOT EXISTS categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
);
"""

CREATE_BOOKS_TABLE = """
CREATE TABLE IF NOT EXISTS books (
    book_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT NOT NULL,
    price_gbp    REAL NOT NULL,
    price_inr    REAL NOT NULL,
    rating       INTEGER NOT NULL,
    in_stock     INTEGER NOT NULL,   -- 0 / 1 (SQLite has no native BOOLEAN)
    stock_count  INTEGER NOT NULL,
    category_id  INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
"""


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open (and implicitly create) the SQLite database file."""
    return sqlite3.connect(db_path)


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the categories and books tables if they don't already exist."""
    conn.execute(CREATE_CATEGORIES_TABLE)
    conn.execute(CREATE_BOOKS_TABLE)
    conn.commit()


def reset_schema(conn: sqlite3.Connection) -> None:
    """
    Drop and recreate both tables. Used at the start of a fresh pipeline
    run so re-running main.py never produces duplicate rows.
    """
    conn.execute("DROP TABLE IF EXISTS books;")
    conn.execute("DROP TABLE IF EXISTS categories;")
    conn.commit()
    create_schema(conn)


def insert_categories(conn: sqlite3.Connection, category_names: list[str]) -> dict[str, int]:
    """
    Insert each unique category name (ignoring duplicates) and return a
    {category_name: category_id} lookup for use when inserting books.
    """
    unique_names = sorted(set(category_names))

    conn.executemany(
        "INSERT OR IGNORE INTO categories (category_name) VALUES (?);",
        [(name,) for name in unique_names],
    )
    conn.commit()

    rows = conn.execute("SELECT category_id, category_name FROM categories;").fetchall()
    return {name: category_id for category_id, name in rows}


def insert_books(conn: sqlite3.Connection, books_df: pd.DataFrame, category_map: dict[str, int]) -> None:
    """Insert every row of `books_df` into the books table."""
    records = []
    for row in books_df.itertuples(index=False):
        records.append(
            (
                row.title,
                float(row.price_gbp),
                float(row.price_inr),
                int(row.rating),
                int(bool(row.in_stock)),
                int(row.stock_count),
                category_map[row.category],
            )
        )

    conn.executemany(
        """
        INSERT INTO books
            (title, price_gbp, price_inr, rating, in_stock, stock_count, category_id)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        records,
    )
    conn.commit()


def load_dataframe_to_db(conn: sqlite3.Connection, books_df: pd.DataFrame) -> None:
    """
    Full load: create a fresh schema, insert categories, then insert books
    linked to the correct category_id.
    """
    reset_schema(conn)
    category_map = insert_categories(conn, books_df["category"].tolist())
    insert_books(conn, books_df, category_map)
    print(f"[database_manager] Loaded {len(books_df)} books across {len(category_map)} categories.")
