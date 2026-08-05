"""
scraper.py

Scrapes book data from https://books.toscrape.com/, a free static site
built specifically for scraping practice.

Architecture (fixed):
    get_page()               -> download raw HTML for a URL
    get_soup()                -> turn HTML into a BeautifulSoup object
    get_categories()          -> list every book category on the site
    get_books_from_category() -> walk a category's pages, collect book URLs
    get_book_details()        -> visit one book page, extract its full record
    scrape_books()             -> master function that wires the above together
"""

import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from constants import (
    BASE_URL,
    MIN_BOOKS,
    MIN_CATEGORIES,
    POLITE_DELAY_SECONDS,
    REQUEST_HEADERS,
    REQUEST_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BACKOFF_SECONDS,
)
from utils import clean_text, convert_gbp_to_inr, parse_availability, parse_price, parse_rating


def get_page(url: str) -> str | None:
    """
    Download the raw HTML for `url`.

    Retries a few times with a short backoff before giving up, since a
    single flaky request shouldn't kill a run that has already visited
    dozens of pages. Returns None (instead of raising) if every attempt
    fails, so callers can decide how to handle a missing page.
    """
    last_error = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            response.encoding = response.apparent_encoding  # avoid mangled "£"
            time.sleep(POLITE_DELAY_SECONDS)
            return response.text
        except requests.RequestException as error:
            last_error = error
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    print(f"[scraper] Giving up on {url} after {REQUEST_RETRIES} attempts: {last_error}")
    return None


def get_soup(html: str) -> BeautifulSoup:
    """Parse raw HTML into a BeautifulSoup object."""
    return BeautifulSoup(html, "html.parser")


def get_categories() -> list[dict]:
    """
    Read the homepage and return every category listed in the left-hand
    sidebar nav.

    Returns:
        [{"name": "Travel", "url": "https://books.toscrape.com/catalogue/category/books/travel_2/index.html"}, ...]
    """
    html = get_page(BASE_URL)
    if not html:
        return []

    soup = get_soup(html)
    nav_links = soup.select("div.side_categories ul.nav-list ul li a")

    categories = []
    for link in nav_links:
        name = clean_text(link.get_text())
        url = urljoin(BASE_URL, link["href"])
        categories.append({"name": name, "url": url})

    return categories


def get_books_from_category(category: dict) -> list[dict]:
    """
    Walk every page of a single category (handling "next" pagination) and
    collect a lightweight reference to every book found.

    Returns:
        [{"book_url": "...", "rating": 3}, ...]
    """
    books = []
    page_url = category["url"]

    while page_url:
        html = get_page(page_url)
        if not html:
            break

        soup = get_soup(html)

        for article in soup.select("article.product_pod"):
            relative_url = article.select_one("h3 a")["href"]
            book_url = urljoin(page_url, relative_url)

            rating_tag = article.select_one("p.star-rating")
            rating_word = rating_tag["class"][1] if rating_tag else "Zero"

            books.append({"book_url": book_url, "rating": parse_rating(rating_word)})

        next_link = soup.select_one("li.next a")
        page_url = urljoin(page_url, next_link["href"]) if next_link else None

    return books


def get_book_details(book_url: str, category_name: str, rating: int) -> dict | None:
    """
    Visit a single book's detail page and extract its full record.

    `rating` is passed in from get_books_from_category() rather than
    re-parsed here, since the catalogue page already gave it to us and the
    detail page encodes it the exact same way (no need to fetch it twice).

    Returns:
        {
            "title": "...", "category": "...", "price_gbp": 45.17,
            "price_inr": 4765.44, "rating": 3, "in_stock": True,
            "stock_count": 19,
        }
        or None if the page could not be read.
    """
    html = get_page(book_url)
    if not html:
        return None

    soup = get_soup(html)

    title = clean_text(soup.select_one("div.product_main h1").get_text())

    price_gbp = parse_price(soup.select_one("p.price_color").get_text())
    price_inr = convert_gbp_to_inr(price_gbp)

    availability_text = soup.select_one("p.availability").get_text()
    in_stock, stock_count = parse_availability(availability_text)

    return {
        "title": title,
        "category": category_name,
        "price_gbp": price_gbp,
        "price_inr": price_inr,
        "rating": rating,
        "in_stock": in_stock,
        "stock_count": stock_count,
    }


def scrape_books(max_categories: int = MIN_CATEGORIES, min_books: int = MIN_BOOKS) -> list[dict]:
    """
    Master function. Workflow:

        read categories
          -> take categories in the order the site lists them
          -> visit every page of each category (pagination handled)
          -> collect every book URL + rating
          -> visit every book and extract its full details
          -> return the flat list of book dictionaries

    Takes at least `max_categories` categories, but if those categories
    together don't reach `min_books` books, keeps adding the next category
    in the list until the minimum is met (or categories run out). This
    keeps the acceptance criteria (>= 60 books across >= 3 categories) safe
    even if the live site's category sizes ever change.
    """
    categories = get_categories()
    if not categories:
        print("[scraper] No categories found - is the site reachable?")
        return []

    all_books: list[dict] = []
    categories_used = 0

    for category in categories:
        if categories_used >= max_categories and len(all_books) >= min_books:
            break

        print(f"[scraper] Scraping category: {category['name']}")
        book_refs = get_books_from_category(category)

        for ref in book_refs:
            details = get_book_details(ref["book_url"], category["name"], ref["rating"])
            if details:
                all_books.append(details)

        categories_used += 1

    print(f"[scraper] Done. {len(all_books)} books across {categories_used} categories.")
    return all_books
