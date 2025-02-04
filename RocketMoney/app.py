import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import re
from collections import deque
import time

# Selenium & webdriver_manager imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ---------------------------------------------------------------------
# 1. Initialize Selenium WebDriver (cached so it's created once)
# ---------------------------------------------------------------------
@st.cache_resource
def init_selenium_driver():
    """
    Creates a headless Selenium Chrome WebDriver using webdriver-manager.
    Caches the driver so it's only created once during the app's lifecycle.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# ---------------------------------------------------------------------
# 2. Scraping Functions
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
    Fetch a page via Selenium (dynamic).
    wait_time: seconds to wait for JavaScript to load.
    Returns the raw HTML.
    """
    driver = init_selenium_driver()
    driver.get(url)
    time.sleep(wait_time)
    html = driver.page_source
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
    Returns a dict with extracted info.
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
        attrs = {k: v for k, v in meta.attrs.items()}
        metas.append(attrs)
    data["meta_tags"] = metas

    # Images
    images = []
    for img in soup.find_all("img"):
        src = img.get("src")
        alt = img.get("alt")
        images.append({"src": src, "alt": alt})
    data["images"] = images

    # Links
    links = []
    for a in soup.find_all("a", href=True):
        link_href = a["href"].strip()
        link_text = a.get_text(strip=True)
        links.append({"href": link_href, "text": link_text})
    data["links"] = links

    # Include original URL
    data["url"] = url
    return data

# ---------------------------------------------------------------------
# 3. BFS Crawler
# ---------------------------------------------------------------------
def bfs_crawl(start_url, max_depth, max_pages, mode, wait_time):
    """
    BFS-based crawler:
      - Follows links within the same domain as start_url.
      - Up to max_depth levels deep.
      - Up to max_pages total pages crawled.
      - mode: "static" or "dynamic"
    Returns a list of dicts, each containing extracted data for one page.
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

        # Fetch HTML using the selected mode
        if mode == "dynamic":
            html = fetch_dynamic(current_url, wait_time)
        else:
            html = fetch_static(current_url)

        # Extract data from the current page
        page_data = extract_data(current_url, html)
        results.append(page_data)
        pages_crawled += 1

        # Update progress (clamped to 100%)
        progress_bar.progress(min(int((pages_crawled / max_pages) * 100), 100))

        if pages_crawled >= max_pages:
            break

        # Parse links for BFS expansion
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
# 4. Streamlit App
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
                try:
                    results = bfs_crawl(
                        start_url, 
                        max_depth=max_depth, 
                        max_pages=max_pages, 
                        mode=mode_choice, 
                        wait_time=wait_time
                    )
                    st.success("Crawling Complete!")
                    
                    st.subheader("Results (JSON)")
                    for i, page_dict in enumerate(results, 1):
                        with st.expander(f"Page {i}: {page_dict.get('url', 'N/A')}"):
                            st.json(page_dict)
                    
                    json_data = json.dumps(results, indent=2)
                    st.download_button(
                        label="Download All Results as JSON",
                        data=json_data,
                        file_name="scraping_results.json",
                        mime="application/json"
                    )
                
                except Exception as e:
                    st.error(f"Scraping failed: {e}")

if __name__ == "__main__":
    main()
