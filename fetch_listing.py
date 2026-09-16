"""Dohvat detalja jednog oglasa."""

import re
from datetime import datetime, timezone
from html import unescape
from urllib.parse import urlparse

from search_jobs import (
    USER_AGENT,
    _active_listing_status,
    _extract_application_deadline,
    _extract_unavailable_after,
    _extract_wage,
    _has_student_evidence,
    _html_to_text,
    _get_listing_response,
    _is_allowed_listing_url,
    _meta_description,
    _normalize,
    _normalize_url,
    _readable_excerpt,
)


def fetch_listing(url):
    """Dohvaca pojedinacni oglas i izdvaja dostupne podatke."""
    if not _is_allowed_listing_url(url):
        return {"error": "URL nije dopusteni pojedinacni oglas."}

    try:
        response = _get_listing_response(url)
    except Exception as exc:
        return {"error": f"Dohvat oglasa nije uspio: {exc}"}

    final_url = _normalize_url(response.url)
    if response.status_code != 200 or not final_url or not _is_allowed_listing_url(final_url):
        return {
            "error": "Stranica nije potvrdila valjani pojedinacni oglas.",
            "status_code": response.status_code,
            "final_url": final_url,
        }

    text = _html_to_text(response.text)
    normalized = _normalize(text)
    active, active_evidence = _active_listing_status(text, response.text)
    # `unavailable_after` je tehnički robots meta-podatak o dostupnosti
    # stranice. Može potvrditi aktivnost, ali nije nužno rok za prijavu i ne
    # smije se korisniku prikazati kao takav.
    deadline = _extract_application_deadline(normalized)
    return {
        "verified": True,
        "active": active,
        "active_evidence": active_evidence,
        "final_url": final_url,
        "source_domain": urlparse(final_url).netloc.lower().removeprefix("www."),
        "status_code": response.status_code,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "title": _title(response.text) or _label_value(text, ("Naziv posla", "Pozicija")),
        "employer": _label_value(text, ("Poslodavac", "Tvrtka")),
        "location": _label_value(text, ("Lokacija", "Mjesto rada")),
        "working_hours": _label_value(text, ("Radno vrijeme", "Sati tjedno", "Broj sati")),
        "application_deadline": deadline.isoformat() if deadline else None,
        "detected_wage_eur_h": _extract_wage(normalized),
        "student_listing": _has_student_evidence(normalized),
        "remote": _contains_any(normalized, ("rad od kuce", "rad od kuće", "remote", "na daljinu")),
        "remote_evidence": _first_evidence(text, ("rad od kuće", "remote", "na daljinu")),
        "weekend_only": _weekend_only(normalized),
        "weekend_evidence": _weekend_only_evidence(text),
        "experience_required": _experience_required(normalized),
        "experience_evidence": _first_evidence(text, _EXPERIENCE_EVIDENCE_MARKERS),
        "content_excerpt": _meta_description(response.text) or _readable_excerpt(text, max_chars=900),
    }


def _title(html):
    for pattern in (
        r'<meta\s+property="og:title"\s+content="([^"]+)"',
        r"<title[^>]*>(.*?)</title>",
    ):
        match = re.search(pattern, html or "", re.I | re.S)
        if match:
            title = re.sub(r"\s+", " ", unescape(match.group(1))).strip()
            if title:
                return title[:250]
    return None


def _label_value(text, labels):
    for label in labels:
        match = re.search(
            rf"{re.escape(label)}\s*:?\s*(.{{2,240}}?)(?=Rok za prijavu|Način prijave|Nacin prijave|Prijave uputiti|📅 Objavljeno|$)",
            text,
            re.I,
        )
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" -:")
            if 2 <= len(value) <= 180:
                return value
    return None


def _contains_any(text, markers):
    return any(_normalize(marker) in text for marker in markers)


def _first_evidence(text, markers):
    normalized = _normalize(text)
    for marker in markers:
        position = normalized.find(_normalize(marker))
        if position >= 0:
            return re.sub(r"\s+", " ", text[max(0, position - 55):position + 125]).strip()
    return None


_NO_EXPERIENCE_MARKERS = (
    "nije potrebno iskustvo",
    "nije potrebno radno iskustvo",
    "nije potrebno prethodno iskustvo",
    "nije potrebno prethodno radno iskustvo",
    "bez iskustva",
    "bez radnog iskustva",
    "bez prethodnog iskustva",
    "bez prethodnog radnog iskustva",
    "iskustvo nije potrebno",
    "prethodno iskustvo nije potrebno",
    "prethodno radno iskustvo nije potrebno",
    "iskustvo nije uvjet",
)

_EXPERIENCE_EVIDENCE_MARKERS = (
    *_NO_EXPERIENCE_MARKERS,
    "potrebno iskustvo",
    "radno iskustvo",
    "iskustvo u",
)


def _experience_required(text):
    text = _normalize(text)
    if _contains_any(text, _NO_EXPERIENCE_MARKERS):
        return False
    if "radno iskustvo" in text or "iskustvo u" in text:
        return True
    return None


def _weekend_only(text):
    """Prepoznaje raspored koji dokazivo ne ukljucuje radne dane."""
    text = _normalize(text)
    explicit_weekend = (
        "iskljucivo vikendom",
        "samo vikendom",
        "subotom i nedjeljom",
        "subota i nedjelja",
        "subotom, nedjeljom",
        "subota/nedjelja",
    )
    weekday_markers = (
        "radnim dan", "ponedjeljak", "utorak", "srijeda", "cetvrtak", "petak",
    )
    if _has_negative_weekend_schedule(text):
        return False
    return _contains_any(text, explicit_weekend) and not _contains_any(text, weekday_markers)


def _has_negative_weekend_schedule(text):
    """Prepoznaje da se subotom/nedjeljom izricito ne radi."""
    return bool(re.search(
        r"(?:ne\s+(?:radimo|radi(?:\s+se)?|radite)|bez\s+rada)\s+"
        r"(?:subot(?:a|om)|nedjelj(?:a|om)|vikend)|"
        r"(?:vikendi?|subote?|nedjelje?)\s+(?:su\s+)?(?:slobodni|neradni)|"
        r"vikend\s+planove",
        text,
    ))


def _weekend_only_evidence(text):
    markers = (
        "isključivo vikendom", "samo vikendom", "subotom i nedjeljom",
        "subota i nedjelja", "subotom, nedjeljom", "subota/nedjelja",
    )
    return _first_evidence(text, markers) if _weekend_only(text) else None
