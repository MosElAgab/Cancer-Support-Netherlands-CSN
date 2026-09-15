"""Small URL helper functions."""

from urllib.parse import urlparse


def get_url_section(url):
    """Return first URL path section, or homepage."""

    path = urlparse(url).path
    parts = path.strip("/").split("/")
    return parts[0] if parts and parts[0] else "homepage"


def url_depth(url):
    """Count URL path depth."""

    path = urlparse(url).path
    return len([part for part in path.split("/") if part])


def is_valid_url(url, base_domain, blocked_sections=None, max_url_depth=6):
    """Apply basic crawl filters."""

    if blocked_sections is None:
        blocked_sections = []

    parsed = urlparse(url)

    if parsed.netloc and parsed.netloc != base_domain:
        return False

    if parsed.fragment:
        return False

    if parsed.query:
        return False

    if url_depth(url) > max_url_depth:
        return False

    section = get_url_section(url)

    if section in blocked_sections:
        return False

    return True
