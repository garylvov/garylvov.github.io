"""Fail deployment if the homepage references missing or invalid WebP images."""

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class ImageSources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "source" and attrs.get("type") == "image/webp":
            for candidate in attrs.get("srcset", "").split(","):
                if candidate.strip():
                    self.urls.add(candidate.split()[0])


def check(site_dir):
    sources = ImageSources()
    sources.feed((site_dir / "index.html").read_text())
    if not sources.urls:
        raise ValueError("Homepage has no responsive WebP sources to validate")

    for url in sorted(sources.urls):
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc:
            raise ValueError(f"Expected a local generated image: {url}")
        image = site_dir / unquote(parsed.path).lstrip("/")
        with image.open("rb") as file:
            header = file.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WEBP":
            raise ValueError(f"Invalid WebP image: {image}")

    print(f"Verified {len(sources.urls)} homepage WebP variants")


if __name__ == "__main__":
    check(Path(sys.argv[1] if len(sys.argv) > 1 else "_site"))
