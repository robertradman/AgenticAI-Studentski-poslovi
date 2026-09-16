"""Spremanje korisnickih preferencija i prethodno prikazanih oglasa."""

import json
import os
import tempfile
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ROOT_DIR = os.path.dirname(__file__)
STATE_PATH = os.path.join(ROOT_DIR, "data", "task_state.json")
MAX_RECENT_VIEWS = 20
TRACKING_QUERY_PARAMETERS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def task_state(action, data=None, state_path=STATE_PATH):
    data = data or {}
    state = _load_state(state_path)

    if action == "get_preferences":
        return {"preferences": state["preferences"]}

    if action == "get_earnings":
        return dict(state["earnings"])

    if action == "set_earnings":
        amount = float(data.get("earned_so_far_eur", 0))
        if amount < 0:
            raise ValueError("earned_so_far_eur ne moze biti negativan.")
        state["earnings"]["earned_so_far_eur"] = amount
        _write_state(state_path, state)
        return dict(state["earnings"])

    if action == "set_preferences":
        preferences = state["preferences"]
        city = data.get("city")
        keywords = data.get("keywords")
        if city is not None:
            preferences["city"] = str(city).strip() or None
        if keywords is not None:
            preferences["keywords"] = _clean_keywords(keywords)
        _write_state(state_path, state)
        return {"preferences": preferences}

    if action == "get_recent_listings":
        limit = _bounded_limit(data.get("limit", 3), MAX_RECENT_VIEWS)
        return {"listings": state["recent_views"][:limit]}

    if action == "get_seen_urls":
        return {"urls": list(state["seen_listings"].keys())}

    if action == "record_listings":
        listings = data.get("listings")
        if not isinstance(listings, list):
            raise ValueError("record_listings ocekuje listu oglasa.")
        recorded = _record_listings(state, listings)
        _write_state(state_path, state)
        return {"recorded": recorded, "recent_views": state["recent_views"][:3]}

    if action == "reset":
        _write_state(state_path, _empty_state())
        return {"reset": True}

    raise ValueError(f"Nepoznata akcija stanja: {action}")


def _empty_state():
    return {
        "version": 1,
        "preferences": {"city": None, "keywords": []},
        "earnings": {"earned_so_far_eur": 0.0},
        "seen_listings": {},
        "recent_views": [],
    }


def _load_state(state_path):
    if not os.path.exists(state_path):
        return _empty_state()
    try:
        with open(state_path, encoding="utf-8") as state_file:
            stored = json.load(state_file)
    except OSError as exc:
        raise ValueError(f"Stanje nije moguce procitati: {exc}") from exc
    except json.JSONDecodeError as exc:
        # Ne prepisuj nečitljivu datoteku praznim stanjem: to bi trajno
        # izgubilo povijest i već viđene oglase.
        raise ValueError("Stanje je osteceno i nije prepisano. Sacuvajte kopiju datoteke prije popravka.") from exc

    state = _empty_state()
    if isinstance(stored, dict):
        state["preferences"].update(stored.get("preferences") or {})
        state["earnings"].update(stored.get("earnings") or {})
        if isinstance(stored.get("seen_listings"), dict):
            state["seen_listings"] = _canonical_seen_listings(stored["seen_listings"])
        if isinstance(stored.get("recent_views"), list):
            state["recent_views"] = _canonical_recent_views(stored["recent_views"])
    return state


def _record_listings(state, listings):
    recorded = 0
    current_urls = []
    for listing in listings:
        if not isinstance(listing, dict):
            continue
        url = canonical_listing_url(listing.get("final_url") or listing.get("url"))
        if not url:
            continue
        safe_listing = {
            key: listing.get(key)
            for key in (
                "title", "url", "final_url", "source_domain", "application_deadline",
                "detected_wage_eur_h", "posted_at", "active", "verified",
            )
        }
        safe_listing["url"] = url
        safe_listing["final_url"] = url
        safe_listing["viewed_at"] = datetime.now(timezone.utc).isoformat()
        state["seen_listings"][url] = safe_listing
        current_urls.append(url)
        recorded += 1

    prior = [
        listing for listing in state["recent_views"]
        if (listing.get("final_url") or listing.get("url")) not in current_urls
    ]
    newest = [state["seen_listings"][url] for url in current_urls]
    state["recent_views"] = (newest + prior)[:MAX_RECENT_VIEWS]
    return recorded


def canonical_listing_url(value):
    """Vraća stabilan identitet oglasa za trajnu deduplikaciju."""
    if not isinstance(value, str):
        return None
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    query = [
        (key, item) for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_QUERY_PARAMETERS
    ]
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, urlencode(query, doseq=True), ""))


def _canonical_seen_listings(listings):
    canonical = {}
    for stored_url, listing in listings.items():
        if not isinstance(listing, dict):
            continue
        url = canonical_listing_url(listing.get("final_url") or listing.get("url") or stored_url)
        if not url:
            continue
        canonical[url] = {**listing, "url": url, "final_url": url}
    return canonical


def _canonical_recent_views(listings):
    recent = []
    seen = set()
    for listing in listings:
        if not isinstance(listing, dict):
            continue
        url = canonical_listing_url(listing.get("final_url") or listing.get("url"))
        if not url or url in seen:
            continue
        seen.add(url)
        recent.append({**listing, "url": url, "final_url": url})
        if len(recent) >= MAX_RECENT_VIEWS:
            break
    return recent


def _write_state(state_path, state):
    directory = os.path.dirname(state_path)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix="state-", suffix=".json", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as state_file:
            json.dump(state, state_file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, state_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _clean_keywords(keywords):
    if isinstance(keywords, str):
        keywords = [keywords]
    if not isinstance(keywords, list):
        raise ValueError("keywords mora biti tekst ili lista tekstova.")
    return [str(keyword).strip() for keyword in keywords if str(keyword).strip()][:8]


def _bounded_limit(value, maximum):
    try:
        return max(1, min(int(value), maximum))
    except (TypeError, ValueError):
        return 3
