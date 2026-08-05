"""
constants.py

Every "magic value" used across the data_pipeline module lives here so that
nothing is hard-coded inside the business-logic files. If the site URL, the
exchange rate, or the database path ever need to change, this is the only
file that should need editing.
"""

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
# Database
# --------------------------------------------------------------------------
DB_PATH: str = "books.db"

# Rating words used on the site map to these integers.
RATING_WORD_TO_INT: dict = {
    "Zero": 0,
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}
