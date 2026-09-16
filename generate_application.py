"""Izrada nacrta motivacijskog pisma za provjereni oglas."""

import os

from fetch_listing import fetch_listing


def generate_application(listing_url, applicant_profile):
    """Izradjuje nacrt motivacijskog pisma na temelju oglasa i profila korisnika."""
    profile = _validate_profile(applicant_profile)
    if not os.environ.get("GEMINI_API_KEY"):
        return {"error": "Za agentic izradu motivacijskog pisma potreban je GEMINI_API_KEY."}
    listing = fetch_listing(listing_url)
    if listing.get("error"):
        return {"error": listing["error"]}
    if not listing.get("verified") or not listing.get("active"):
        return {"error": "Motivacijsko pismo se moze izraditi samo za aktivan provjereni oglas."}

    prompt = """Napisi kratak nacrt motivacijskog pisma na hrvatskom jeziku.
Koristi samo informacije iz oglasa i profila kandidata. Nemoj izmisljati iskustvo,
vjestine, obrazovanje ni kontaktne podatke. Ako podatak nije naveden, nemoj ga
spominjati. Pismo treba imati pozdrav, dva kratka odlomka i zavrsetak.

Oglas:
Naslov: {title}
Poslodavac: {employer}
Lokacija: {location}
Opis: {excerpt}

Profil kandidata:
{profile}
""".format(
        title=listing.get("title") or "nije naveden",
        employer=listing.get("employer") or "nije naveden",
        location=listing.get("location") or "nije navedena",
        excerpt=listing.get("content_excerpt") or "nije dostupan",
        profile=profile,
    )

    from google import genai
    try:
        response = genai.Client(api_key=os.environ["GEMINI_API_KEY"]).models.generate_content(
            # Isti zadani model kao glavni agent
            model=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
            contents=prompt,
        )
    except Exception as exc:
        return {"error": f"Izrada nacrta nije uspjela: {exc}"}

    letter = (response.text or "").strip()
    if not letter:
        return {"error": "Model nije vratio tekst motivacijskog pisma."}
    return {
        "listing_url": listing.get("final_url") or listing_url,
        "listing_title": listing.get("title"),
        "letter": letter,
        "note": "Ovo je nacrt. Prije slanja provjerite sadrzaj i dopunite osobne podatke.",
    }


def _validate_profile(profile):
    if not isinstance(profile, str) or len(profile.strip()) < 30:
        raise ValueError("Potrebno je navesti barem kratak profil kandidata.")
    return profile.strip()[:4000]
