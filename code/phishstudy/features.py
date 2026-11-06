"""Engineered message features and the normalised text used by the text model.

All quantities are computed from the subject, the visible body text and the
structural counts returned by ``parse.parse_message``. The same code path is
applied to every collection.
"""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"'()\[\]{}]+", re.I)
# Addresses, including the domain-less placeholder that Phishing Pot substitutes for recipients.
MAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\bphishing@pot\b", re.I)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
# Lexicon inherited from the pre-execution protocol (fixed before any data were read).
URGENT = frozenset("urgent immediately verify verification suspend suspended account password login "
                   "confirm update limited expire expires security".split())
CLICK_TEXT = re.compile(r"\b(click|here|login|log in|sign in|update|verify|confirm)\b", re.I)
# Identifiers of the collectors themselves (inbox owner, honeypot placeholder, corpus maintainer),
# deleted from the model text so that they cannot act as collection labels.
COLLECTOR_TOKENS = re.compile(r"\b(jose|nazario|monkey\.org|phishing@pot|phishing pot|jmason|"
                              r"spamassassin\.taint\.org|taint\.org)\b", re.I)

FORMAT_FEATURES = ["characters", "words", "unique_words", "line_count", "mean_word_length",
                   "upper_fraction", "digit_fraction", "punctuation_fraction", "nonascii_fraction",
                   "html_body", "html_only", "attachments"]
URL_FEATURES = ["url_count", "unique_url_hosts", "ip_url_count", "max_host_dots", "anchor_count",
                "mismatched_anchors", "action_link_text", "script_or_form"]
LEXICAL_FEATURES = ["stopwords", "urgent_words", "email_addresses"]
URGENCY_FEATURES = ["urgent_words"]
FEATURES = FORMAT_FEATURES + LEXICAL_FEATURES + URL_FEATURES


def _host(url: str) -> str:
    try:
        u = url if "://" in url else "http://" + url
        return (urlsplit(u).hostname or "").lower().strip(".")
    except ValueError:
        return ""


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return bool(re.fullmatch(r"\d+(?:\.\d+){3}", host))


def model_text(subject: str, body: str) -> str:
    """Subject and body with URLs and addresses replaced and collector identifiers removed."""
    text = f"{subject}\n{body}"
    text = MAIL_RE.sub(" emailtoken ", text)
    text = URL_RE.sub(" urltoken ", text)
    text = COLLECTOR_TOKENS.sub(" ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def engineered(subject: str, body: str, parsed: dict) -> dict:
    text = f"{subject}\n{body}"
    words = WORD_RE.findall(text)
    lower = [w.lower() for w in words]
    n = max(len(text), 1)
    # Distinct URL strings from the visible text and from anchor targets, so that a link repeated in
    # the plain and HTML alternatives of one message is counted once.
    urls = sorted(set(URL_RE.findall(text)) |
                  {h for h, _ in parsed["anchors"] if h.lower().startswith(("http", "www"))})
    hosts = {_host(u) for u in urls} - {""}
    mismatched = 0
    action = 0
    for href, label in parsed["anchors"]:
        shown = URL_RE.search(label or "")
        if shown and _host(shown.group(0)) and _host(href) and _host(shown.group(0)) != _host(href):
            mismatched += 1
        if CLICK_TEXT.search(label or ""):
            action += 1
    lines = [ln for ln in body.split("\n") if ln.strip()]
    return dict(
        characters=len(text), words=len(words), unique_words=len(set(lower)),
        line_count=len(lines), mean_word_length=float(np.mean([len(w) for w in words])) if words else 0.0,
        upper_fraction=sum(c.isupper() for c in text) / n, digit_fraction=sum(c.isdigit() for c in text) / n,
        punctuation_fraction=sum((not c.isalnum()) and (not c.isspace()) for c in text) / n,
        nonascii_fraction=sum(ord(c) > 127 for c in text) / n,
        html_body=int(parsed["has_html"]), html_only=int(parsed["has_html"] and not parsed["has_plain"]),
        attachments=int(parsed["attachments"]),
        stopwords=sum(w in ENGLISH_STOP_WORDS for w in lower), urgent_words=sum(w in URGENT for w in lower),
        email_addresses=len(MAIL_RE.findall(text)),
        url_count=len(urls), unique_url_hosts=len(hosts), ip_url_count=sum(_is_ip(_host(u)) for u in urls),
        max_host_dots=max((h.count(".") for h in hosts), default=0), anchor_count=len(parsed["anchors"]),
        mismatched_anchors=mismatched, action_link_text=action,
        script_or_form=int(bool(parsed["script"] or parsed["form"])),
    )


def canonical(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def template_text(text: str) -> str:
    """Matching representation only; never used as a model input."""
    s = canonical(text)
    s = re.sub(r"\d+", "0", s)
    return s
