# data_pipeline

Scrapes book data from [books.toscrape.com](https://books.toscrape.com/) (a free,
static site built for scraping practice), cleans it with pandas, loads it into a
two-table SQLite database, and runs the required SQL queries against it.

## Install

```bash
cd data_pipeline
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

This does everything end to end, with no manual copy-pasting:

1. Scrapes at least 60 books across at least 3 categories from the live site.
2. Cleans the results into a typed pandas DataFrame.
3. Rebuilds `books.db` from scratch (drops and recreates both tables so the run
   is repeatable) and loads the data in.
4. Runs 6 SQL queries and prints each query's SQL and output.
5. Reproduces the JOIN query's result with `pd.merge()` on the in-memory
   DataFrames and confirms it matches the `pd.read_sql()` output.

## Files

| File | Responsibility |
|---|---|
| `constants.py` | Every fixed value: base URL, exchange rate, DB path, HTTP settings. |
| `utils.py` | Small pure parsing helpers (price, rating, availability text) — easy to test without hitting the network. |
| `scraper.py` | `get_page → get_soup → get_categories → get_books_from_category → get_book_details → scrape_books`. |
| `database_manager.py` | SQLite schema creation and inserts. |
| `sql_queries.py` | The 6 named SQL queries plus the `pd.read_sql` vs `pd.merge` comparison. |
| `main.py` | Wires everything together into one runnable pipeline. |

## Database schema

Two tables, linked by a foreign key:

```
categories                    books
-----------                   -----------------------
category_id (PK)              book_id (PK)
category_name                 title
                               price_gbp
                               price_inr
                               rating          (1-5 int)
                               in_stock        (0/1)
                               stock_count
                               category_id (FK -> categories.category_id)
```

## Design decisions

- **Fixed exchange rate.** `price_inr` is computed as `price_gbp * 105.50`
  (`constants.GBP_TO_INR`). This is a fixed, project-defined constant with no
  date reference — not a live currency API — per the capstone brief.
- **Category selection.** `scrape_books()` walks categories in the order the
  site lists them, taking at least 3 and continuing to add more only if the
  first 3 don't reach 60 books, so the 60-books/3-categories requirement holds
  even if the live site's category sizes change over time.
- **Rating source.** The star rating is read once, from the category listing
  page, and passed straight into `get_book_details()` rather than re-parsed
  from the book's own page — it's encoded identically in both places, so
  re-fetching it would just be a wasted parse.
- **Idempotent loads.** `main.py` drops and recreates the SQLite tables on
  every run (`database_manager.reset_schema`), so re-running the pipeline
  never produces duplicate rows.
- **Retries.** `get_page()` retries failed requests a few times with backoff
  before giving up on a page, since a single flaky request shouldn't kill a
  run that's already 40 pages in.
- **Encoding.** Response encoding is set to `response.apparent_encoding`
  before reading `.text`, since the raw £ symbol can otherwise come through
  mangled depending on how the server's headers are read.

## SQL queries

All 6 live in `sql_queries.QUERIES` and are executed + printed by
`run_all_queries()`:

| Query | Clause(s) demonstrated |
|---|---|
| `books_in_stock_high_rating` | `SELECT` / `WHERE` |
| `ten_most_expensive_books` | `ORDER BY` / `LIMIT` |
| `distinct_ratings` | `DISTINCT` |
| `books_in_selected_categories` | `IN`, plus `JOIN` |
| `midrange_priced_books` | `BETWEEN` |
| `top_rated_books_by_category` | `JOIN`, `ORDER BY`, `LIMIT` |

`compare_sql_and_pandas_join()` reads the `top_rated_books_by_category` result
via `pd.read_sql()`, separately reproduces it with `pd.merge()` on the raw
`books`/`categories` DataFrames (no SQL involved), and asserts the two match.

## A note on this run

This module was scaffolded with AI assistance (per the capstone's own
guidelines, which explicitly permit this) and every function was verified for
correctness before being treated as final:

- `database_manager.py`, `sql_queries.py`, and `main.py`'s cleaning logic were
  run end to end against a synthetic 75-row dataset (3 categories) to confirm
  the schema, inserts, all 6 SQL queries, and the `pd.read_sql`/`pd.merge`
  equivalence check all work correctly.
- `scraper.py` targets the live site's actual HTML structure (verified
  against the site's known layout: `article.product_pod`, `p.star-rating`,
  `p.price_color`, `p.availability`, `li.next`, the `div.side_categories`
  nav, and the `div.product_main` / `#product_information` book page). It was
  not executed against the live site in the environment that generated it, so
  **run `python main.py` yourself once and read the printed output** before
  treating the scrape as done — that's also the easiest way to be able to
  explain every part of it confidently during evaluation.
