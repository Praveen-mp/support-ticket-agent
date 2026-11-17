import pytest

from tests.fakes import load_kb
from ticket_agent.domain.faq import FaqKnowledgeBase, UrlRef, extract_urls


def test_parses_all_twelve_articles():
    kb = load_kb()
    assert kb.article_ids == tuple(f"F{i:02d}" for i in range(1, 13))
    f05 = kb.get("F05")
    assert f05 is not None
    assert f05.title == "Plans and prices"
    assert "€4 per active employee" in f05.text
    # The rules section after the last article is not part of F12.
    f12 = kb.get("F12")
    assert f12 is not None
    assert "client's rules" not in f12.text


def test_known_urls_come_from_the_faq():
    kb = load_kb()
    assert kb.is_known_url(UrlRef("app.roosterly.nl", "/login"))
    assert kb.is_known_url(UrlRef("roadmap.roosterly.nl", ""))
    assert kb.is_known_url(UrlRef("app.roosterly.nl", ""))
    assert not kb.is_known_url(UrlRef("roadmap.roosterly.nl", "/vote"))
    assert not kb.is_known_url(UrlRef("roosterly-support.com", ""))


def test_extract_urls_skips_email_domains_and_trailing_punctuation():
    refs = extract_urls("Mail hulp@roosterly.nl of ga naar https://www.Status.roosterly.nl/.")
    assert refs == [UrlRef("status.roosterly.nl", "")]


def test_rejects_faq_without_articles():
    with pytest.raises(ValueError):
        FaqKnowledgeBase("# Empty help centre")
