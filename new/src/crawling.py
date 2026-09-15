"""Crawl source websites and collect discovered links."""

import time
from collections import deque
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.url_utils import get_url_section, is_valid_url


def extract_links(
    url,
    depth,
    base_url,
    base_domain,
    headers,
    blocked_sections=None,
    max_url_depth=6,
):
    """Extract valid links from one source page."""

    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        link_text = a.get_text(" ", strip=True)

        if len(link_text) < 3:
            continue

        if link_text.lower() in ["volgende", "vorige", "next", "previous"]:
            continue

        full_url = urljoin(base_url, href)

        if not is_valid_url(
            full_url,
            base_domain=base_domain,
            blocked_sections=blocked_sections,
            max_url_depth=max_url_depth,
        ):
            continue

        links.append(
            {
                "source_page": url,
                "depth": depth,
                "link_text": link_text,
                "href": href,
                "full_url": full_url,
                "url_section": get_url_section(full_url),
            }
        )

    return links


def crawl_site(
    start_url,
    base_url,
    base_domain,
    source_organisation,
    blocked_sections=None,
    headers=None,
    max_depth=2,
    max_pages=50,
    max_url_depth=6,
    sleep_seconds=1,
):
    """Crawl one website and return discovered link observations."""

    if blocked_sections is None:
        blocked_sections = []

    if headers is None:
        headers = {"User-Agent": "Mozilla/5.0"}

    visited = set()
    queue = deque([(start_url, 0)])
    all_links = []

    while queue and len(visited) < max_pages:
        current_url, depth = queue.popleft()

        if current_url in visited:
            continue

        visited.add(current_url)
        print(f"Scraping depth {depth}: {current_url}")

        try:
            links = extract_links(
                url=current_url,
                depth=depth,
                base_url=base_url,
                base_domain=base_domain,
                headers=headers,
                blocked_sections=blocked_sections,
                max_url_depth=max_url_depth,
            )

            all_links.extend(links)

            if depth < max_depth:
                for link in links:
                    next_url = link["full_url"]

                    if next_url not in visited:
                        queue.append((next_url, depth + 1))

            time.sleep(sleep_seconds)

        except Exception as e:
            print(f"Failed: {current_url}")
            print(e)

    df = pd.DataFrame(all_links)

    if not df.empty:
        df["source_organisation"] = source_organisation

        column_order = [
            "source_organisation",
            "source_page",
            "depth",
            "link_text",
            "href",
            "full_url",
            "url_section",
        ]
        df = df[column_order]

    return df
