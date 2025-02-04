import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from collections import deque
import time
import re

# Import Playwright's sync API for dynamic scraping
from playwright.sync_api import sync_playwright

# ---------------------------------------------------------------------
# 1. Scraping Functions
# ---------------------------------------------------------------------
def fetch_static(url, headers=None):
    """Fetch a page using requests (static). Returns raw HTML."""
    if not headers:
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
    wait_time: seconds to wait for JavaScript to load.
    Returns the raw HTML.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        # Increase the timeout if needed (milliseconds)
        page.goto(url, timeout=15000)
        page.wait_for_timeout(wait_time * 1000)  # wait_time in ms
        html = page.content()
        browser.close()
        return html

def parse_html(html):
    """Convert HTML string to a BeautifulSoup object."""
    return BeautifulSoup(html, "html.parser")

def extract_data(url, html):
    """
    Extract advanced data from a single page:
      - Title
      - Headings (h1, h2, h3)
      - Meta tags
      - Images (src, alt)
      - Links (href and text)
    Returns a dictionary with the extracted info.
    """
    soup = parse_html(html)
    data = {}

    # Title
    title_tag = soup.find("title")
    data["title"] = title_tag.get_text(strip=True) if title_tag else None

    # Headings
    headings = []
    for tag_name in ["h1", "h2", "h3"]:
        for tag in soup.find_all(tag_name):
            headings.append({
                "tag": tag_name,
                "text": tag.get_text(strip=True)
            })
    data["headings"] = headings

    # Meta tags
    metas = []
    for meta in soup.find_all("meta"):
        metas.append(dict(meta.attrs))
    data["meta_tags"] = metas

    # Images
    images = []
    for img in soup.find_all("img"):
        images.append({
            "src": img.get("src"),
            "alt": img.get("alt")
        })
    data["images"] = images

    # Links
    links = []
    for a in soup.find_all("a", href=True):
        links.append({
            "href": a["href"].strip(),
            "text": a.get_text(strip=True)
        })
    data["links"] = links

    # Include the original URL
    data["url"] = url

    return data

# ---------------------------------------------------------------------
# 2. BFS Crawler
# ---------------------------------------------------------------------
def bfs_crawl(start_url, max_depth, max_pages, mode, wait_time):
    """
    BFS-based crawler:
      - Follows links within the same domain as start_url.
      - Crawls up to max_depth levels and max_pages pages.
      - mode: "static" (requests) or "dynamic" (Playwright).
    Returns a list of dictionaries (one per page) with extracted data.
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

        progress = min(int((pages_crawled / max_pages) * 100), 100)
        progress_bar.progress(progress)

        if pages_crawled >= max_pages:
            break

        soup = parse_html(html)
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            next_link = urljoin(current_url, href)
            next_domain = urlparse(next_link).netloc.lower()
            if next_domain == domain and next_link not in visited:
                visited.add(next_link)
                queue.append((next_link, depth + 1))

    return results

# ---------------------------------------------------------------------
# 3. Streamlit App UI
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Ultra Advanced Web Scraper", layout="wide")
    st.title("Ultra Advanced Web Scraper")

    st.sidebar.header("Scraping Parameters")
    start_url = st.sidebar.text_input("Start URL (e.g., https://example.com)")
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

            for i, page in enumerate(results, 1):
                with st.expander(f"Page {i}: {page.get('url', 'N/A')}"):
                    st.json(page)

            json_data = json.dumps(results, indent=2)
            st.download_button(
                label="Download All Results as JSON",
                data=json_data,
                file_name="scraping_results.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
