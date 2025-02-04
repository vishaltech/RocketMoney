import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from collections import deque
import time
import re
import os
import subprocess

# Import Playwright’s sync API
from playwright.sync_api import sync_playwright

# =============================================================================
# Helper: Ensure Playwright Browsers Are Installed
# =============================================================================
def ensure_playwright_browsers_installed():
    """
    Check if Playwright’s Chromium browser executable exists.
    If not, run 'playwright install' to download the necessary browsers.
    """
    # Typically, Playwright caches its browsers in ~/.cache/ms-playwright
    cache_dir = os.path.expanduser("~/.cache/ms-playwright")
    if not os.path.exists(cache_dir):
        st.info("Playwright browsers not found. Installing now...")
        try:
            subprocess.run(["playwright", "install"], check=True)
        except Exception as e:
            st.error(f"Error during Playwright browser installation: {e}")

# =============================================================================
# Helper: Fetch Functions (Static & Dynamic)
# =============================================================================
def fetch_static(url, headers=None):
    """Fetch a page using requests (static). Returns raw HTML."""
    if headers is None:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/58.0.3029.110 Safari/537.3"
            )
        }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.text

def fetch_dynamic(url, wait_time=3):
    """
    Fetch a page using Playwright (dynamic).
    Automatically ensures browsers are installed.
    wait_time: seconds to wait for JavaScript to load.
    Returns raw HTML.
    """
    ensure_playwright_browsers_installed()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        # Navigate to the URL (15-second timeout)
        page.goto(url, timeout=15000)
        page.wait_for_timeout(wait_time * 1000)  # wait_time in milliseconds
        html = page.content()
        browser.close()
        return html

def parse_html(html):
    """Convert an HTML string to a BeautifulSoup object."""
    return BeautifulSoup(html, "html.parser")

# =============================================================================
# Helper: Extract Key Table Data from the Page
# =============================================================================
def extract_key_table(soup):
    """
    Find all table elements in the page and return the one with the most rows.
    The table is returned as a dictionary containing headers and rows.
    """
    tables = soup.find_all("table")
    key_table = None
    max_rows = 0
    for table in tables:
        tbody = table.find("tbody")
        if tbody:
            rows = tbody.find_all("tr")
            if len(rows) > max_rows:
                max_rows = len(rows)
                key_table = table
    if key_table:
        # Attempt to extract headers from <thead>; fallback to first row if missing.
        headers = []
        thead = key_table.find("thead")
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all("th")]
        else:
            first_row = key_table.find("tr")
            if first_row:
                headers = [cell.get_text(strip=True) for cell in first_row.find_all(["td", "th"])]

        # Extract row data from <tbody>; if missing, use all rows.
        data_rows = []
        tbody = key_table.find("tbody")
        rows = tbody.find_all("tr") if tbody else key_table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            cell_texts = [cell.get_text(strip=True) for cell in cells]
            if headers and len(cell_texts) == len(headers):
                data_rows.append(dict(zip(headers, cell_texts)))
            else:
                data_rows.append(cell_texts)
        return {"headers": headers, "rows": data_rows}
    return None

# =============================================================================
# Helper: Extract Data from a Page
# =============================================================================
def extract_data(url, html):
    """
    Extract various data from a page:
      - Title, headings, meta tags, images, links, and (if present) key table data.
    Returns a dictionary of the extracted data.
    """
    soup = parse_html(html)
    data = {}

    # Title
    title_tag = soup.find("title")
    data["title"] = title_tag.get_text(strip=True) if title_tag else None

    # Headings (h1, h2, h3)
    headings = []
    for tag_name in ["h1", "h2", "h3"]:
        for tag in soup.find_all(tag_name):
            headings.append({"tag": tag_name, "text": tag.get_text(strip=True)})
    data["headings"] = headings

    # Meta tags
    metas = []
    for meta in soup.find_all("meta"):
        metas.append(dict(meta.attrs))
    data["meta_tags"] = metas

    # Images
    images = []
    for img in soup.find_all("img"):
        images.append({"src": img.get("src"), "alt": img.get("alt")})
    data["images"] = images

    # Links
    links = []
    for a in soup.find_all("a", href=True):
        links.append({"href": a["href"].strip(), "text": a.get_text(strip=True)})
    data["links"] = links

    # Extract key table data (if any)
    key_table = extract_key_table(soup)
    if key_table:
        data["key_table"] = key_table
    else:
        data["key_table"] = "No table data found"

    # Include the original URL
    data["url"] = url

    return data

# =============================================================================
# BFS Crawler Function
# =============================================================================
def bfs_crawl(start_url, max_depth, max_pages, mode, wait_time):
    """
    Crawl the given start_url using a breadth-first search (BFS) approach.
      - Only follows links within the same domain.
      - Limits the crawl by max_depth and max_pages.
      - mode: "static" (requests) or "dynamic" (Playwright).
    Returns a list of dictionaries containing extracted data for each page.
    """
    domain = urlparse(start_url).netloc.lower()
    visited = set()
    queue = deque()
    queue.append((start_url, 0))
    visited.add(start_url)

    results = []
    pages_crawled = 0

    progress_bar = st.progress(0)

    while queue:
        current_url, depth = queue.popleft()
        if depth > max_depth:
            break

        try:
            if mode == "dynamic":
                html = fetch_dynamic(current_url, wait_time)
            else:
                html = fetch_static(current_url)
        except Exception as e:
            st.error(f"Failed to fetch {current_url}: {e}")
            continue

        page_data = extract_data(current_url, html)
        results.append(page_data)
        pages_crawled += 1

        # Update progress bar (as a percentage)
        progress_bar.progress(min(int((pages_crawled / max_pages) * 100), 100))

        if pages_crawled >= max_pages:
            break

        # Parse links and add new ones from the same domain
        soup = parse_html(html)
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            next_link = urljoin(current_url, href)
            next_domain = urlparse(next_link).netloc.lower()
            if next_domain == domain and next_link not in visited:
                visited.add(next_link)
                queue.append((next_link, depth + 1))

    return results

# =============================================================================
# Streamlit App UI
# =============================================================================
def main():
    st.set_page_config(page_title="Ultra Advanced Web Scraper", layout="wide")
    st.title("Ultra Advanced Web Scraper")

    st.sidebar.header("Scraping Parameters")
    start_url = st.sidebar.text_input("Start URL (e.g., https://stockanalysis.com/stocks/)")
    max_depth = st.sidebar.number_input("Max Depth", min_value=0, max_value=10, value=1)
    max_pages = st.sidebar.number_input("Max Pages", min_value=1, max_value=1000, value=5)
    mode_choice = st.sidebar.selectbox("Scraping Mode", ["static", "dynamic"])
    wait_time = st.sidebar.slider("Dynamic Wait (seconds)", 1, 10, 3)

    if st.sidebar.button("Start Scraping"):
        if not start_url:
            st.error("Please provide a start URL.")
        else:
            with st.spinner("Crawling in progress..."):
                results = bfs_crawl(start_url, max_depth, max_pages, mode_choice, wait_time)
            st.success("Crawling Complete!")
            st.subheader("Results (JSON)")

            # Display each page’s data in an expander
            for i, page in enumerate(results, 1):
                with st.expander(f"Page {i}: {page.get('url', 'N/A')}"):
                    st.json(page)

            # Offer all results as a downloadable JSON file
            json_data = json.dumps(results, indent=2)
            st.download_button(
                label="Download All Results as JSON",
                data=json_data,
                file_name="scraping_results.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
