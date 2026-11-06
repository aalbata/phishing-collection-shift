"""Safe reading of RFC 822 messages from mbox, tar and zip archives.

Archives are read member by member in memory; nothing is extracted to disk, no
link is dereferenced, no HTML is rendered by a browser engine, and attachments
are counted but never decoded into the text representation.
"""
from __future__ import annotations

import hashlib
import mailbox
import re
import tarfile
import warnings
import zipfile
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path, PurePosixPath

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

MAX_MESSAGE_BYTES = 10 * 2**20
MAX_HTML_CHARS = 2_000_000
MAX_TEXT_CHARS = 100_000
SPAMASSASSIN_NAME = re.compile(r"^\d{5}\.[0-9a-f]{32}$")
ENRON_NAME = re.compile(r"^\d+\.$")
PARSER = BytesParser(policy=policy.default)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()


def iter_archive(path: Path, kind: str, collection: str, select=None):
    """Yield (locator, raw_bytes, member_meta) for every candidate message.

    ``select(locator, meta)``, when given, is evaluated before a tar member is read; members it
    rejects are skipped without reading their content.
    """
    path = Path(path)
    if kind == "mbox":
        box = mailbox.mbox(str(path), factory=None, create=False)
        try:
            for key in box.iterkeys():
                raw = box.get_bytes(key, from_=False)
                yield f"mbox:{key}", raw, {}
        finally:
            box.close()
    elif kind == "tar":
        with tarfile.open(path, "r|*") as tf:
            for member in tf:
                if not member.isfile() or member.size > MAX_MESSAGE_BYTES:
                    continue
                parts = PurePosixPath(member.name).parts
                name = parts[-1]
                if collection == "spamassassin_ham":
                    if not SPAMASSASSIN_NAME.match(name):
                        continue
                    meta = {}
                elif collection == "enron":
                    if not ENRON_NAME.match(name) or len(parts) < 4 or parts[0] != "maildir":
                        continue
                    meta = {"mailbox_user": parts[1], "folder": "/".join(parts[2:-1])}
                else:
                    raise ValueError(f"no tar layout registered for {collection}")
                if select is not None and not select(member.name, meta):
                    continue
                stream = tf.extractfile(member)
                if stream is not None:
                    yield member.name, stream.read(), meta
    elif kind == "zip":
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                name = PurePosixPath(info.filename)
                if info.is_dir() or name.suffix.lower() != ".eml" or "email" not in name.parts:
                    continue
                if info.file_size > MAX_MESSAGE_BYTES:
                    continue
                with zf.open(info) as stream:
                    yield "/".join(name.parts[1:]), stream.read(), {}
    else:
        raise ValueError(kind)


def _decode_part(part) -> str:
    try:
        content = part.get_content()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        return content
    except Exception:
        payload = part.get_payload(decode=True)
        if payload is None:
            raw = part.get_payload()
            return raw if isinstance(raw, str) else ""
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except LookupError:
            return payload.decode("latin-1", errors="replace")


def _body_parts(part):
    """Inline text leaves only; attachment subtrees and attached messages are pruned."""
    if part.get_content_disposition() == "attachment" or part.get_content_type() == "message/rfc822":
        return
    if part.is_multipart():
        for child in part.iter_parts():
            yield from _body_parts(child)
    else:
        yield part


def _count_attachments(msg) -> int:
    n = 0
    for part in msg.walk():
        if part.is_multipart():
            continue
        disp = part.get_content_disposition()
        ctype = part.get_content_type()
        if disp == "attachment" or ctype == "message/rfc822" or not ctype.startswith("text/"):
            n += 1
    return n


def html_to_text(markup: str) -> tuple[str, dict]:
    """Visible text and structural counts of an HTML part (parsed, never rendered)."""
    soup = BeautifulSoup(markup[:MAX_HTML_CHARS], "html.parser")
    info = dict(
        anchors=[(a.get("href", "") or "", a.get_text(" ", strip=True)) for a in soup.find_all("a", href=True)],
        script=bool(soup.find("script")) or "javascript:" in markup.lower(),
        form=bool(soup.find("form")),
    )
    for node in soup(["script", "style", "noscript", "head", "title"]):
        node.decompose()
    text = soup.get_text("\n")
    lines = [re.sub(r"[ \t\r\f\v ]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln), info


def parse_message(raw: bytes) -> dict:
    """Decode one message into subject, visible body text and structural counts."""
    msg = PARSER.parsebytes(raw)
    try:
        subject = str(msg.get("Subject", "") or "")
    except Exception:
        subject = ""
    subject = re.sub(r"\s+", " ", subject).strip()
    try:
        sender = parseaddr(str(msg.get("From", "") or ""))[1].lower()
    except Exception:
        sender = ""
    year = None
    try:
        if msg.get("Date"):
            year = parsedate_to_datetime(str(msg.get("Date"))).year
    except Exception:
        year = None
    plain, html_parts = [], []
    for part in _body_parts(msg):
        ctype = part.get_content_type()
        if ctype == "text/plain":
            plain.append(_decode_part(part))
        elif ctype == "text/html":
            html_parts.append(_decode_part(part))
    anchors, script, form = [], False, False
    html_text = []
    for markup in html_parts:
        text, info = html_to_text(markup)
        html_text.append(text)
        anchors += info["anchors"]
        script = script or info["script"]
        form = form or info["form"]
    # One representation per message: the plain alternative when present, else the HTML text.
    if any(p.strip() for p in plain):
        body = "\n".join(plain)
        body = re.sub(r"\r\n?", "\n", body)
        body = "\n".join(re.sub(r"[ \t\f\v ]+", " ", ln).strip() for ln in body.split("\n"))
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
    else:
        body = "\n".join(html_text).strip()
    body = body.replace("\x00", "")[:MAX_TEXT_CHARS]
    return dict(subject=subject[:1000], body=body, year=year, sender=sender, has_html=bool(html_parts),
                has_plain=any(p.strip() for p in plain), anchors=anchors, script=script, form=form,
                attachments=_count_attachments(msg))
