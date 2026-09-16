"""Pretraga i osnovna provjera oglasa za posao.

Pretraga koristi javno dostupne podatke s portala za zapošljavanje. Dobiveni
rezultati sadrže naslov, poveznicu i kratki opis, dok se detalji pojedinog oglasa
dohvaćaju zasebno. Za lokalno testiranje dostupan je i offline način rada.
"""

import json
import os
import re
import threading
import unicodedata
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import monotonic
from html import unescape
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from urllib.parse import urlencode, urljoin, urlparse

ROOT_DIR = os.path.dirname(__file__)
FIXTURE_PATHS = [
    os.path.join(ROOT_DIR, "fixtures", "sample_results.json"),
    os.path.join(ROOT_DIR, "sample_results.json"),
]

DOMENE = ["studentposao.hr", "mojposao.hr", "studentski-servis.hr", "posao.hr"]
EXTERNAL_DOMENE = ["mojposao.hr", "studentski-servis.hr", "posao.hr"]
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)
# Svaki kandidat se mora otvoriti i potvrditi kao aktivan. Četiri kandidata s
# primarnog portala i dva s preostalih dopustenih izvora daju dovoljno izbora
# agentu, a izbjegavaju desetke HTTP zahtjeva za jedan korisnicki upit.
MAX_LIVE_RESULTS = 6
MAX_STUDENTPOSAO_CANDIDATES = 4
MAX_EXTERNAL_CANDIDATES = 2
HTTP_TIMEOUT_SECONDS = 5
LISTING_CACHE_TTL_SECONDS = 300
MAX_LISTING_CACHE_SIZE = 128
MAX_VERIFICATION_WORKERS = 8
_listing_cache = OrderedDict()
_listing_cache_lock = threading.Lock()
STOP_WORDS = {
    "a",
    "ako",
    "aktivan",
    "aktivna",
    "aktivne",
    "aktivni",
    "da",
    "h",
    "hr",
    "i",
    "ili",
    "ima",
    "koji",
    "koje",
    "koja",
    "mi",
    "na",
    "najmanje",
    "ne",
    "od",
    "oglas",
    "oglase",
    "oglasi",
    "po",
    "posao",
    "poslove",
    "poslova",
    "pronadi",
    "pronađi",
    "s",
    "sa",
    "student",
    "studentska",
    "studentski",
    "studentske",
    "studentskih",
    "su",
    "trazim",
    "tražim",
    "u",
    "za",
}
INACTIVE_MARKERS = [
    "ovaj oglas nije aktivan",
    "prijava na njega nije moguca",
    "prijava na njega nije moguća",
    "oglas nije aktivan",
    "istekao je rok prijave",
]


def search_jobs(query, city=None, min_wage=None, offline=None):
    if offline is None:
        offline = os.environ.get("AGENT_OFFLINE") == "1"

    if offline:
        return _search_offline(query, city, min_wage)
    return _search_live(query, city, min_wage)


def _build_query(query, city, min_wage):
    parts = [query]
    if city:
        parts.append(city)
    if min_wage:
        parts.append(f"{min_wage} EUR/h")
    site_filter = " OR ".join(f"site:{d}" for d in DOMENE)
    parts.append(f"({site_filter})")
    return " ".join(parts)


def _search_live(query, city, min_wage):
    seen_urls = set()
    search_errors = []
    results = _search_studentposao(query, city, min_wage)
    for item in results:
        seen_urls.add(item["url"])

    # StudentPosao je primarni izvor, ali nije jedini. Jedna ograničena
    # pretraga pokriva preostala tri portala za svaki upit neovisno o
    # tome je li StudentPosao već dao rezultat.
    try:
        from ddgs import DDGS
    except ModuleNotFoundError:
        return results[:MAX_LIVE_RESULTS]

    with DDGS(timeout=8) as ddgs:
        search_query = _build_external_query(query, city, min_wage)
        try:
            raw_results = ddgs.text(search_query, region="hr-hr", max_results=MAX_EXTERNAL_CANDIDATES * 3)
        except Exception as exc:
            search_errors.append(f"{search_query}: {exc}")
            raw_results = []

        candidates = []
        for r in raw_results:
            item = {
                "title": r.get("title"),
                "url": r.get("href"),
                "snippet": r.get("body"),
                "search_query": search_query,
            }
            normalized_url = _normalize_url(item["url"])
            if not normalized_url or normalized_url in seen_urls:
                continue
            if not _is_allowed_listing_url(normalized_url):
                continue
            if urlparse(normalized_url).netloc.lower().removeprefix("www.") not in EXTERNAL_DOMENE:
                continue

            item["url"] = normalized_url
            candidates.append(item)
            if len(candidates) >= MAX_EXTERNAL_CANDIDATES:
                break

        for item, verification in _verify_candidates(candidates):
            if not verification["verified"]:
                continue
            item.update(verification)
            if not _matches_live_filters(item, query, city, min_wage):
                continue

            seen_urls.add(item["url"])
            results.append(item)
            if len(results) >= MAX_LIVE_RESULTS:
                break

    fatal_errors = [error for error in search_errors if "No results found" not in error]
    if not results and fatal_errors:
        raise RuntimeError("Live pretraga nije uspjela: " + " | ".join(fatal_errors[:2]))

    return results


def _build_external_query(query, city, min_wage):
    """Jedna tražilišna pretraga samo za preostale domenske izvore."""
    if _is_remote_query(query):
        terms = "studentski online remote posao"
    elif _is_weekend_query(query):
        terms = "studentski posao vikendom"
    else:
        terms = "studentski posao " + (_studentposao_keyword(query) or "")
    parts = [terms.strip()]
    if city:
        parts.append(city)
    if min_wage is not None:
        parts.append(f"{min_wage} EUR/h")
    parts.append("(" + " OR ".join(f"site:{domain}" for domain in EXTERNAL_DOMENE) + ")")
    return " ".join(parts)


def _build_live_queries(query, city, min_wage):
    base = _build_query(query, city, min_wage)
    city_part = f" {city}" if city else ""
    wage_part = f" {min_wage} EUR/h" if min_wage else ""
    return [
        base,
        f"{query}{city_part}{wage_part} site:studentposao.hr/oglas/",
        f"{query}{city_part}{wage_part} studentski ugovor site:mojposao.hr/posao/",
        f"{query}{city_part}{wage_part} studentski posao site:www.posao.hr/oglasi/",
        f"{query}{city_part}{wage_part} student site:studentski-servis.hr",
    ]


def _search_studentposao(query, city, min_wage):
    import requests

    # StudentPosao podržava vlastiti q/city filter. Jedan popis zamjenjuje
    # dva gotovo jednaka endpointa i uklanja potrebu za širokom pretragom.
    list_urls = _studentposao_list_urls(query, city, min_wage)
    candidates = []
    seen = set()

    for list_url in list_urls:
        try:
            response = requests.get(
                list_url,
                headers={"User-Agent": USER_AGENT, "Accept-Language": "hr-HR,hr;q=0.9,en;q=0.8"},
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            if response.status_code != 200:
                continue
        except Exception:
            continue

        for item in _parse_studentposao_cards(response.text, list_url):
            if item["url"] in seen:
                continue
            if not _studentposao_card_matches(item, city, min_wage, query=query):
                continue
            seen.add(item["url"])
            item["listed_in_active_feed"] = True
            if _is_remote_query(query):
                item["listed_in_remote_feed"] = True
            candidates.append(item)

    results = []
    # Ne otvaraj desetke pojedinačnih oglasa. Stranica je već filtrirana po
    # upitu, gradu i po potrebi satnici; mali broj kandidata ostavlja dovoljno
    # izbora za odluku agenta bez nepotrebnog mrežnog opterećenja.
    for batch in _batches(candidates[:MAX_STUDENTPOSAO_CANDIDATES], MAX_VERIFICATION_WORKERS):
        for item, verification in _verify_candidates(batch):
            if not verification["verified"]:
                continue
            _apply_active_feed_evidence(item, verification)
            item.update(verification)
            if not _matches_live_filters(item, query, city, min_wage):
                continue
            results.append(item)
            if len(results) >= MAX_LIVE_RESULTS:
                return results

    return results


def _parse_studentposao_cards(html, base_url):
    items = []
    for chunk in html.split('<div class="job-card')[1:]:
        # Izmedu href i aria-label trenutno stoji class atribut. Ne oslanjaj
        # se na redoslijed HTML atributa, jer bi to preskocilo sve kartice.
        href_match = re.search(
            r'<a\b(?=[^>]*\bhref=["\']([^"\']+)["\'])(?=[^>]*\baria-label=["\']([^"\']+)["\'])[^>]*>',
            chunk,
            re.S,
        )
        if not href_match:
            continue
        href, title = href_match.groups()
        company = _strip_html_match(
            re.search(r'<p class="text-gray-500[^"]*">\s*(.*?)\s*</p>', chunk, re.S)
        )
        location = _strip_html_match(
            re.search(r'<p class="text-gray-400[^"]*">\s*📍\s*(.*?)\s*</p>', chunk, re.S)
        )
        wage = _strip_html_match(
            re.search(r'<div class="[^"]*font-bold text-primary[^"]*">\s*(.*?)\s*</div>', chunk, re.S)
        )
        published = _strip_html_match(
            re.search(r'<span class="text-xs text-gray-400[^"]*">\s*(.*?)\s*</span>', chunk, re.S)
        )
        category = _extract_card_category(chunk)
        url = _normalize_url(urljoin(base_url, href))
        if not url:
            continue
        snippet_parts = [
            f"Tvrtka: {company}" if company else "",
            f"Lokacija: {location}" if location else "",
            f"Satnica: {wage}" if wage else "",
            f"Objavljeno: {published}" if published else "",
            "Studentski posao na StudentPosao.hr",
        ]
        items.append(
            {
                "title": unescape(title.strip()),
                "url": url,
                "snippet": " | ".join(part for part in snippet_parts if part),
                "search_query": base_url,
                "published": published,
                "posted_at": _parse_published_date(published),
                "category": category,
            }
        )
    return items


def _studentposao_list_urls(query, city, min_wage=None):
    """Vraća jednu ili dvije uske pretrage kada upit ima dva odvojena uvjeta."""
    keywords = [_studentposao_keyword(query)]
    if _is_it_query(query) and _is_internship_query(query):
        keywords = ["praksa", "IT"]
    urls = []
    for keyword in keywords:
        url = _studentposao_list_url(query, city, min_wage, keyword=keyword)
        if url not in urls:
            urls.append(url)
    return urls


def _studentposao_list_url(query, city, min_wage=None, keyword=None):
    """Gradi jedan ograničeni URL službene pretrage StudentPosao portala."""
    params = {}
    if city:
        params["city"] = city
    if min_wage is not None:
        params["min_salary"] = min_wage
    remote = _is_remote_query(query)
    if remote:
        params["is_remote"] = "true"
    if keyword is None:
        keyword = _studentposao_keyword(query, remote)
    if keyword:
        params["q"] = keyword
    return "https://studentposao.hr/?" + urlencode(params)


def _studentposao_keyword(query, remote=False):
    """Vraća samo pojam zanimanja, nikad pomoćne riječi iz LLM parafraze."""
    normalized = _normalize(query)
    if remote:
        # Službeni Online posao filter već pokriva i "online" i "remote".
        # Dodatni q=remote bi pogrešno isključio online oglase bez te riječi.
        return None
    if _is_weekend_query(query):
        return "vikend"
    if _is_internship_query(query):
        return "praksa"
    if _is_it_query(query):
        return "IT"
    category_keywords = {
        "programer": "programer",
        "developer": "developer",
        "informat": "informatika",
        "konobar": "konobar",
        "sanker": "sanker",
        "kuhar": "kuhar",
    }
    for marker, keyword in category_keywords.items():
        if marker in normalized:
            return keyword
    # Za opće upite (grad, satnica, iskustvo, vrijeme rada) ne šalji q. Oni
    # nisu zanimanje i ne smiju suziti službeni popis.
    return None


def _studentposao_card_matches(item, city, min_wage, query=""):
    """Jeftino filtrira kartice prije dohvaćanja njihovih detalja."""
    evidence = _normalize(" ".join(str(item.get(key) or "") for key in ("title", "snippet")))
    if city and _normalize(city) not in evidence:
        return False
    if _is_it_query(query) and item.get("category") and item["category"] != "it i digitalno":
        return False
    if min_wage is not None:
        wage = _extract_wage(evidence) or _extract_card_wage(evidence)
        # Službeni min_salary filter je već primijenjen na izvoru. Neke kartice
        # prikazuju samo "10 €" bez oznake /h, pa ih detaljna provjera ne smije
        # prerano odbaciti zbog formatiranja prikaza.
        if wage is not None and wage < float(min_wage):
            return False
    return True


def _extract_card_category(html):
    """Izdvaja vidljivu kategoriju s kartice prije skupog dohvata detalja."""
    card_text = _normalize(_strip_html_match(re.search(r"(.+)", html, re.S)) or "")
    categories = (
        "it i digitalno", "ugostiteljstvo", "turizam", "trgovina i prodaja",
        "skladiste i logistika", "fizicki poslovi", "proizvodnja i tehnicki",
        "ciscenje i odrzavanje", "dostava i voznja", "administracija i ured",
        "call centar i podrska", "promocije i eventi", "instrukcije i rad s djecom",
    )
    for category in categories:
        if category in card_text:
            return category
    return None


def _strip_html_match(match):
    if not match:
        return None
    text = re.sub(r"<[^>]+>", " ", match.group(1))
    return unescape(re.sub(r"\s+", " ", text).strip())


def _search_offline(query, city, min_wage):
    # Jednostavno filtriranje lokalnih primjera za rad bez interneta.
    fixture_path = _existing_fixture_path()
    with open(fixture_path, encoding="utf-8") as f:
        listings = json.load(f)

    query_words = _meaningful_words(query)
    results = []
    for listing in listings:
        text = _normalize(listing["title"] + " " + listing["snippet"])
        if city and _normalize(city) not in text:
            continue
        if min_wage is not None:
            wage = _extract_wage(text)
            if wage is None or wage < float(min_wage):
                continue
        if query_words and not any(word in text for word in query_words):
            continue
        results.append(listing)
    return results


def _normalize_url(url):
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return parsed._replace(fragment="").geturl()


def _is_allowed_listing_url(url):
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.lower()

    if host == "mojposao.hr":
        return path.startswith("/posao/") and len(path.strip("/").split("/")) >= 2

    if host == "posao.hr":
        return path.startswith("/oglasi/") and len(path.strip("/").split("/")) >= 2

    if host == "studentski-servis.hr":
        return "oglas" in path and len(path.strip("/").split("/")) >= 2

    if host == "studentposao.hr":
        return path.startswith("/oglas/") and len(path.strip("/").split("/")) >= 2

    return False


def _verify_listing(url):
    try:
        response = _get_listing_response(url)
        final_url = _normalize_url(response.url)
        verified = response.status_code == 200 and bool(final_url) and _is_allowed_listing_url(final_url)
        full_text = _html_to_text(response.text) if verified else ""
        active, active_evidence = _active_listing_status(full_text, response.text)
        # Tehnički `unavailable_after` služi isključivo za potvrdu aktivnosti.
        # Nije dokaz stvarnog roka prijave i zato ga ne izlažemo kao rok.
        deadline = _extract_application_deadline(_normalize(full_text))
        return {
            "verified": verified,
            "active": verified and active,
            "active_evidence": active_evidence if verified else None,
            "application_deadline": deadline.isoformat() if deadline else None,
            "source_domain": urlparse(final_url).netloc.lower().removeprefix("www.") if final_url else None,
            "status_code": response.status_code,
            "final_url": final_url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "content_excerpt": _meta_description(response.text) or _readable_excerpt(full_text),
        }
    except Exception as exc:
        return _verification_error(url, exc)


def _verify_candidates(candidates):
    """Provjerava više oglasa paralelno, a vraća ih istim redoslijedom."""
    if not candidates:
        return []

    verifications = [None] * len(candidates)
    with ThreadPoolExecutor(max_workers=min(MAX_VERIFICATION_WORKERS, len(candidates))) as executor:
        futures = {
            executor.submit(_verify_listing, item["url"]): index
            for index, item in enumerate(candidates)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                verifications[index] = future.result()
            except Exception as exc:
                verifications[index] = _verification_error(candidates[index]["url"], exc)
    return list(zip(candidates, verifications))


def _get_listing_response(url):
    """Vraća svježi odgovor iz kratkotrajnog cachea ili ga dohvaća s mreže."""
    cache_key = _normalize_url(url) or url
    now = monotonic()
    with _listing_cache_lock:
        cached = _listing_cache.get(cache_key)
        if cached and now - cached[0] < LISTING_CACHE_TTL_SECONDS:
            _listing_cache.move_to_end(cache_key)
            return cached[1]
        _listing_cache.pop(cache_key, None)

    response = _request_listing(cache_key)
    final_url = _normalize_url(response.url)
    with _listing_cache_lock:
        _listing_cache[cache_key] = (now, response)
        if final_url:
            _listing_cache[final_url] = (now, response)
        while len(_listing_cache) > MAX_LISTING_CACHE_SIZE:
            _listing_cache.popitem(last=False)
    return response


def _request_listing(url):
    import requests

    return requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "hr-HR,hr;q=0.9,en;q=0.8"},
        timeout=HTTP_TIMEOUT_SECONDS,
        allow_redirects=True,
    )


def _clear_listing_cache():
    """Prazni cache; koristi se u automatiziranim testovima."""
    with _listing_cache_lock:
        _listing_cache.clear()


def _verification_error(url, exc):
    return {
        "verified": False,
        "active": False,
        "active_evidence": None,
        "application_deadline": None,
        "source_domain": None,
        "status_code": None,
        "final_url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "verification_error": str(exc),
        "content_excerpt": "",
    }


def _batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _matches_live_filters(item, query, city, min_wage):
    raw_evidence = " ".join(
        value or ""
        for value in [
            item.get("title"),
            item.get("snippet"),
            item.get("content_excerpt"),
            item.get("category"),
        ]
    )
    if re.search(r"nedostaje:\s*student", _normalize(raw_evidence)):
        return False

    evidence = _normalize(
        raw_evidence
    )

    if not item.get("active"):
        return False

    if _requires_student_listing(query) and not _has_student_evidence(evidence, item.get("source_domain")):
        return False

    query_words = _meaningful_words(query)
    if query_words and not any(_word_matches_evidence(word, evidence) for word in query_words):
        if item.get("listed_in_remote_feed") and _is_remote_query(query):
            return True
        if item.get("listed_in_active_feed") and _is_weekend_query(query):
            return True
        return False

    if city and _normalize(city) not in evidence:
        return False

    if min_wage is not None:
        wage = _extract_wage(evidence)
        if wage is None and str(item.get("source_domain") or "").lower().removeprefix("www.") == "studentposao.hr":
            wage = _extract_card_wage(evidence)
        if wage is None or wage < float(min_wage):
            return False
        item["detected_wage_eur_h"] = wage

    return True


def _requires_student_listing(query):
    return "student" in _normalize(query)


def _has_student_evidence(evidence, source_domain=None):
    # StudentPosao.hr je namjenski portal za studentske oglase. Za ostale
    # dopuštene portale i dalje zahtijevamo izričit tekstualni dokaz.
    if str(source_domain or "").lower().removeprefix("www.") == "studentposao.hr":
        return True
    student_markers = [
        "studentski ugovor",
        "studentski posao",
        "student/ica",
        "studentica",
        "studenti",
        "studente",
        "za studente",
    ]
    return any(marker in evidence for marker in student_markers)


def _apply_active_feed_evidence(item, verification):
    """Dopunjuje samo neodredenu provjeru aktivnosti aktivnim popisom portala."""
    if not item.get("listed_in_active_feed") or verification.get("active"):
        return
    if verification.get("active_evidence") != "Nema pozitivnog dokaza da je oglas aktivan.":
        return
    verification["active"] = True
    verification["active_evidence"] = "Oglas je upravo naveden na izravnom aktivnom popisu StudentPosao.hr."


def _word_matches_evidence(word, evidence):
    if word in evidence:
        return True
    stem = _light_stem(word)
    return len(stem) >= 4 and stem in evidence


def _light_stem(word):
    for suffix in ("ima", "om", "em", "og", "ih", "oj", "a", "e", "i", "u"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._skip = False
        self._parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            data = data.strip()
            if data:
                self._parts.append(data)

    def text(self):
        return re.sub(r"\s+", " ", " ".join(self._parts)).strip()


def _html_to_text(html):
    parser = _TextExtractor()
    parser.feed(html or "")
    return parser.text()


def _active_listing_status(text, raw_html=""):
    unavailable_after = _extract_unavailable_after(raw_html)
    if unavailable_after:
        if unavailable_after >= date.today():
            return True, f"Stranica je oznacena kao dostupna do {unavailable_after.isoformat()}."
        return False, f"Stranica je oznacena kao dostupna do {unavailable_after.isoformat()}, sto je proslo."

    normalized = _normalize(text)
    if any(_normalize(marker) in normalized for marker in INACTIVE_MARKERS):
        return False, "Pronaden tekst da oglas nije aktivan."

    due_date = _extract_application_deadline(normalized)
    if due_date:
        if due_date >= date.today():
            return True, f"Rok prijave je {due_date.isoformat()}."
        return False, f"Rok prijave je prosao: {due_date.isoformat()}."

    if re.search(r"\b\d+\s+dana?\s+do\s+isteka\b", normalized):
        match = re.search(r"\b\d+\s+dana?\s+do\s+isteka\b", normalized)
        return True, match.group(0)

    if "danas istjece" in normalized or "danas istječe" in normalized:
        return True, "Danas istjece."

    return False, "Nema pozitivnog dokaza da je oglas aktivan."


def _extract_application_deadline(text):
    match = re.search(r"prijava do:?\s*(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})", text)
    if not match:
        return None
    day, month, year = (int(part) for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _extract_unavailable_after(raw_html):
    match = re.search(r"unavailable_after:\s*(\d{4})-(\d{2})-(\d{2})", raw_html or "")
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _slug(text):
    normalized = _normalize(text)
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return normalized


def _parse_published_date(value):
    """Pretvara samo nedvosmislene datume i relativne oznake s liste oglasa."""
    if not value:
        return None
    normalized = _normalize(value)
    # Trenutne kartice StudentPosao.hr koriste i `DD-MM-GGGG`, dok drugi
    # portali često koriste točkasti zapis. Oba su nedvosmisleni datumi.
    exact = re.search(r"(\d{1,2})\s*[.-]\s*(\d{1,2})\s*[.-]\s*(\d{4})", normalized)
    if exact:
        day, month, year = (int(part) for part in exact.groups())
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None
    # 1 dan, 2-4 dana, 5 dana; isti obrazac za 21 dan / 22 dana.
    days_ago = re.search(r"prije\s+(\d+)\s+dan(?:a)?\b", normalized)
    if days_ago:
        return (date.today() - timedelta(days=int(days_ago.group(1)))).isoformat()
    if re.search(r"\bjucer\b", normalized):
        return (date.today() - timedelta(days=1)).isoformat()
    # StudentPosao pise "prije 2 sata" / "prije 5 sati", ne samo "sat" ili "h".
    if (
        "danas" in normalized
        or re.search(r"prije\s+\d+\s+(?:sati|sata|sat|h)\b", normalized)
        or re.search(r"prije\s+\d+\s+(?:minutu|minuta|minute|min)\b", normalized)
    ):
        return date.today().isoformat()
    return None


def _readable_excerpt(text, max_chars=500):
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    noise_phrases = [
        "Pretraži poslove",
        "Profili poslodavaca",
        "Kalkulator plaće",
        "Vijesti i savjeti",
        "Za poslodavce",
        "Postavke kolačića",
    ]
    for phrase in noise_phrases:
        cleaned = cleaned.replace(phrase, "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]


def _meta_description(raw_html):
    match = re.search(
        r'<meta\s+name="description"\s+content="([^"]+)"',
        raw_html or "",
        re.I,
    )
    if not match:
        return None
    return unescape(match.group(1).strip())


def _existing_fixture_path():
    for path in FIXTURE_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Nedostaje offline fixture. Ocekivano: "
        + " ili ".join(FIXTURE_PATHS)
    )


def _normalize(text):
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in text if not unicodedata.combining(char))


def _meaningful_words(query):
    normalized = _normalize(query)
    words = re.findall(r"[a-z0-9]+", normalized)
    return [
        word
        for word in words
        if len(word) > 2 and word not in {_normalize(stop_word) for stop_word in STOP_WORDS}
    ]


def _extract_wage(text):
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:eur|eura|€)\s*/?\s*(?:h|sat)\b", text)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def _extract_card_wage(text):
    """Cita satnicu s kartice StudentPosao, ukljucujuci "10 €" i "10 eura"."""
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:€|eur|eura)(?:\s|$)", text)
    return float(match.group(1).replace(",", ".")) if match else None


def _is_remote_query(query):
    normalized = _normalize(query)
    return any(marker in normalized for marker in ("remote", "online", "rad od kuce", "na daljinu"))


def _is_weekend_query(query):
    normalized = _normalize(query)
    return any(marker in normalized for marker in ("vikend", "subotom", "nedjeljom"))


def _is_it_query(query):
    normalized = _normalize(query)
    return bool(re.search(r"\bit\b", normalized)) or any(
        marker in normalized for marker in ("programer", "developer", "informat", "softver", "software")
    )


def _is_internship_query(query):
    normalized = _normalize(query)
    return any(marker in normalized for marker in ("praksa", "praksi", "praktik", "internship", "trainee"))
