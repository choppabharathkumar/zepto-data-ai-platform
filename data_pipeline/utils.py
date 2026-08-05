"""
utils.py

Small, pure helper functions used while parsing scraped HTML into clean
Python values. Keeping these separate from scraper.py makes them easy to
unit-test on their own (no network calls needed) and keeps scraper.py
focused on *fetching and orchestrating* rather than *string parsing*.
"""

import re

from constants import GBP_TO_INR, RATING_WORD_TO_INT


def clean_text(text: str) -> str:
    """Strip surrounding whitespace/newlines from scraped text."""
    return text.strip() if text else ""


def parse_price(raw_price: str) -> float:
    """
    Convert a raw price string like '£53.74' (or 'Â£53.74' if the page was
    decoded with the wrong encoding) into a float: 53.74

    Any character that is not a digit or a decimal point is stripped out,
    which makes this robust to whichever currency symbol/encoding shows up.
    """
    cleaned = re.sub(r"[^\d.]", "", raw_price or "")
    return round(float(cleaned), 2) if cleaned else 0.0


def convert_gbp_to_inr(price_gbp: float) -> float:
    """Apply the fixed project exchange rate (constants.GBP_TO_INR)."""
    return round(price_gbp * GBP_TO_INR, 2)


def parse_rating(rating_class: str) -> int:
    """
    The site encodes the star rating as a CSS class, e.g.
        <p class="star-rating Three">
    `rating_class` is the *second* class word ("Three"). Convert it to an
    int using constants.RATING_WORD_TO_INT. Unknown values default to 0
    rather than raising, so a single unexpected page never crashes the run.
    """
    return RATING_WORD_TO_INT.get(rating_class, 0)


def parse_availability(raw_text: str) -> tuple[bool, int]:
    """
    The site's availability text looks like:
        'In stock (22 available)'
    or occasionally just 'In stock' / 'Out of stock' with no count.

    Returns (in_stock: bool, stock_count: int).
    """
    text = clean_text(raw_text)
    in_stock = "in stock" in text.lower()

    match = re.search(r"\((\d+)\s*available\)", text)
    stock_count = int(match.group(1)) if match else 0

    return in_stock, stock_count
