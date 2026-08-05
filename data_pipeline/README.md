# data_pipeline

Scrapes book data from [books.toscrape.com](https://books.toscrape.com/) (a free,
static site built for scraping practice), cleans it with pandas, loads it into a
two-table SQLite database, runs the required SQL queries, and saves every
intermediate artefact so the pipeline is fully auditable.

---

## Install

```bash
# From the repo root
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r data_pipeline/requirements.txt
```

Or from inside `data_pipeline/`:

```bash
cd data_pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# From the repo root (recommended)
python data_pipeline/main.py

# Or from inside the subdirectory
cd data_pipeline && python main.py
```

This does everything end to end, with no manual copy-pasting:

| Stage | Action | Output |
|---|---|---|
| 1 | Scrape ≥ 60 books across ≥ 3 categories | in-memory list |
| 2 | Save raw records | `data/raw/books_raw.csv` |
| 3 | Clean & type-validate with pandas | in-memory DataFrame |
| 4 | Save cleaned records | `data/processed/books_clean.csv` |
| 5 | Load into SQLite (two-table PK/FK schema) | `books.db` |
| 6 | Execute 6 SQL queries — print + save | `sql/*.sql`, `outputs/query_outputs/*.txt` |
| 7 | Reproduce JOIN with `pd.merge()` and confirm it matches `pd.read_sql()` | printed comparison |

---

## Directory layout

```
data_pipeline/
├── data/
│   ├── raw/
│   │   └── books_raw.csv          ← raw scraped records (pre-cleaning)
│   └── processed/
│       └── books_clean.csv        ← typed, deduplicated records
├── sql/
│   ├── books_in_stock_high_rating.sql
│   ├── ten_most_expensive_books.sql
│   ├── distinct_ratings.sql
│   ├── books_in_selected_categories.sql
│   ├── midrange_priced_books.sql
│   └── top_rated_books_by_category.sql
├── outputs/
│   └── query_outputs/
│       └── *.txt                  ← one file per query result
├── books.db                       ← SQLite database
├── constants.py
├── utils.py
├── scraper.py
├── database_manager.py
├── sql_queries.py
├── main.py
└── requirements.txt
```

## Source files

| File | Responsibility |
|---|---|
| `constants.py` | Every fixed value: base URL, exchange rate, directory paths, DB path, HTTP settings. |
| `utils.py` | Small pure parsing helpers (price, rating, availability text) — easy to test without hitting the network. |
| `scraper.py` | `get_page → get_soup → get_categories → get_books_from_category → get_book_details → scrape_books`. |
| `database_manager.py` | SQLite schema creation and inserts. |
| `sql_queries.py` | The 6 named SQL queries plus `pd.read_sql` vs `pd.merge` comparison, plus `.sql` file and `.txt` output saving. |
| `main.py` | Wires all 7 pipeline stages together into one runnable script. |

---

## Database schema

Two tables, linked by a foreign key:

```
categories                    books
-----------                   -----------------------
category_id (PK)              book_id (PK)
category_name                 title
                               price_gbp
                               price_inr
                               rating          (1–5 int)
                               in_stock        (0/1)
                               stock_count
                               category_id (FK → categories.category_id)
```

---

## SQL queries

All 6 live in `sql_queries.QUERIES` and are executed + printed by `run_all_queries()`.
Each query is also saved as a `.sql` file and its result as a `.txt` file.

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
`books`/`categories` DataFrames (no SQL join at all), and asserts the two match.

---

## Design decisions

- **Fixed exchange rate.** `price_inr` is computed as `price_gbp × 105.50`
  (`constants.GBP_TO_INR`). This is a fixed, project-defined constant with no
  date reference — not a live currency API — per the capstone brief.
- **Category selection.** `scrape_books()` walks categories in the order the
  site lists them, taking at least 3 and continuing to add more only if the
  first 3 don't reach 60 books, so the 60-books/3-categories requirement holds
  even if the live site's category sizes change over time.
- **Rating source.** The star rating is read once from the category listing
  page and passed straight into `get_book_details()` rather than re-parsed
  from the book's own page — it's encoded identically in both places, so
  re-fetching it would be a wasted parse.
- **Idempotent loads.** `main.py` drops and recreates the SQLite tables on
  every run (`database_manager.reset_schema`), so re-running the pipeline
  never produces duplicate rows.
- **Path anchoring.** All output paths are anchored to `constants.py`'s
  location via `Path(__file__).parent`, so the pipeline writes to the correct
  directories regardless of which working directory `python` is invoked from.
- **Retries.** `get_page()` retries failed requests a few times with backoff
  before giving up on a page, since a single flaky request shouldn't kill a
  run that's already 40 pages in.
- **Encoding.** Response encoding is set to `response.apparent_encoding`
  before reading `.text`, since the raw £ symbol can otherwise come through
  mangled depending on how the server's headers are read.
