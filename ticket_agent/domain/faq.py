"""The client's help centre: the only knowledge the agent may answer from."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_ARTICLE_HEADING = re.compile(r"^## (F\d{2}) · (.+)$", re.MULTILINE)
_SECTION_END = re.compile(r"^(## |---)", re.MULTILINE)
_URL = re.compile(r"\b(?:https?://)?((?:[a-z0-9-]+\.)+[a-z]{2,})(/[^\s)\"'<>]*)?", re.IGNORECASE)


@dataclass(frozen=True)
class FaqArticle:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class UrlRef:
    host: str
    path: str


def extract_urls(text: str) -> list[UrlRef]:
    """Finds host names and URLs, ignoring the domain part of email addresses."""
    refs: list[UrlRef] = []
    for match in _URL.finditer(text):
        if match.start() > 0 and text[match.start() - 1] == "@":
            continue
        host = match.group(1).lower().removeprefix("www.")
        path = (match.group(2) or "").rstrip(".,;:!?").rstrip("/")
        refs.append(UrlRef(host=host, path=path.lower()))
    return refs


class FaqKnowledgeBase:
    def __init__(self, markdown: str) -> None:
        self._markdown = markdown
        self._articles = self._parse_articles(markdown)
        if not self._articles:
            raise ValueError("FAQ contains no articles in the '## Fxx · Title' format")
        self._urls = frozenset(extract_urls(markdown))

    @classmethod
    def from_file(cls, path: Path) -> FaqKnowledgeBase:
        return cls(path.read_text(encoding="utf-8"))

    @property
    def full_text(self) -> str:
        return self._markdown

    @property
    def article_ids(self) -> tuple[str, ...]:
        return tuple(self._articles)

    def get(self, article_id: str) -> FaqArticle | None:
        return self._articles.get(article_id)

    def articles_for(self, article_ids: tuple[str, ...]) -> list[FaqArticle]:
        return [self._articles[i] for i in article_ids if i in self._articles]

    def is_known_url(self, ref: UrlRef) -> bool:
        """A URL is grounded if the FAQ names it exactly, or names its host and it has no path."""
        if ref in self._urls:
            return True
        return not ref.path and any(known.host == ref.host for known in self._urls)

    @staticmethod
    def _parse_articles(markdown: str) -> dict[str, FaqArticle]:
        articles: dict[str, FaqArticle] = {}
        for heading in _ARTICLE_HEADING.finditer(markdown):
            body_start = heading.end()
            end = _SECTION_END.search(markdown, body_start)
            body = markdown[body_start : end.start() if end else len(markdown)].strip()
            article_id, title = heading.group(1), heading.group(2).strip()
            articles[article_id] = FaqArticle(id=article_id, title=title, text=body)
        return articles
