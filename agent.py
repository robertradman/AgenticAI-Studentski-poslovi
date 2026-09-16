"""Glavni program za pretragu studentskih poslova."""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import uuid
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlparse

from estimate_earnings import estimate_earnings
from fetch_listing import fetch_listing
from generate_application import generate_application
from search_jobs import search_jobs
from task_state import canonical_listing_url, task_state

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MAX_STEPS = 15
MAX_RETRIES = 2
# Jedan agentski zadatak treba stati u: pretragu, odabir najrelevantnijih
# oglasa, njihove provjere i zavrsni odgovor. Veci broj turnova samo mnozi
# Gemini zahtjeve kada model pokusa otvoriti cijeli popis kandidata.
MAX_MODEL_TURNS = 6
MAX_LLM_SEARCH_CALLS = 2
MAX_LLM_LISTING_FETCH_CALLS = 3
MAX_LLM_EARNINGS_CALLS = 3
MAX_COMPACT_LISTINGS = 4
MIN_MODEL_CALL_INTERVAL_SECONDS = float(os.environ.get("GEMINI_MIN_CALL_INTERVAL_SECONDS", "0"))
LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "execution_log.jsonl")
KNOWN_CITIES = [
    "Osijek", "Zagreb", "Split", "Rijeka", "Zadar", "Pula", "Varazdin",
    "Slavonski Brod", "Dubrovnik",
]
STOP_WORDS = {
    "a", "ako", "aktivan", "aktivna", "aktivne", "aktivni", "da", "h", "i", "ili",
    "je", "koji", "koje", "koja", "mi", "na", "najmanje", "ne", "od", "oglas",
    "oglase", "oglasi", "po", "posao", "poslove", "poslova", "pronadi", "pronađi",
    "s", "sa", "student", "studentski", "studentske", "studentskih", "su", "u", "za",
}

LLM_SYSTEM_PROMPT = """Ti si AI agent za pretragu studentskih poslova.
Odgovaraj na hrvatskom jeziku i za svaku cinjenicu o oglasu koristi iskljucivo
rezultate pozvanih alata. Nemoj izmisljati oglase, satnice, rokove ili podatke o
kandidatu.

Radi u petlji: nakon svakog rezultata procijeni jesu li uvjeti zadovoljeni i
odaberi sljedeci alat. Nakon search_jobs odaberi samo oglase koje namjeravas
navesti: provjeri najvise tri najrelevantnija oglasa alatom fetch_listing, a ne
cijeli popis rezultata. Za usporedbu tri oglasa provjeri upravo ta tri oglasa.
Ako prva pretraga nema rezultata, obavezno probaj jos jednu pretragu s
drugacijim, srodnim kljucnim rijecima prije zavrsnog odgovora. Kada imas dovoljno
provjerenih podataka, daj zavrsni odgovor bez dodatnih poziva. U zavrsnom
odgovoru smijes navesti samo poveznice oglasa koje je fetch_listing potvrdio kao
aktivne. Kada korisnik trazi usporedbu, koristi samo provjerene oglase i jasno
reci ako nedostaje podatak. Za izracun koristi estimate_earnings, a za preference
i prethodno videne oglase koristi task_state.

Za motivacijsko pismo korisnik mora dostaviti profil te URL ili nedvosmislen
naslov oglasa. Ako je dan naslov, najprije ga pretrazi i odaberi samo jedan
aktivan provjeren oglas. Nacrt se ne salje nikome.
U zavrsnom odgovoru kratko navedi rezultat i provjerene poveznice, bez opisa
unutarnjeg nacina razmisljanja.
"""

FUNCTION_DECLARATIONS = [
    {
        "name": "search_jobs",
        "description": "Pretrazuje aktivne oglase za studentske poslove i prakse.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "city": {"type": "string"},
                "min_wage": {"type": "number"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_listing",
        "description": "Dohvaca i provjerava detalje pojedinacnog oglasa prema URL-u.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "estimate_earnings",
        "description": "Racuna procijenjenu zaradu i odnos prema pragovima.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "hourly_wage": {"type": "number"},
                "hours_per_week": {"type": "number"},
                "weeks": {"type": "number"},
                "earned_so_far": {"type": "number"},
            },
            "required": ["hourly_wage", "hours_per_week"],
        },
    },
    {
        "name": "task_state",
        "description": "Cita ili sprema korisnicke preference i prethodno prikazane oglase.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "get_preferences", "set_preferences", "get_earnings", "set_earnings",
                        "get_recent_listings", "get_seen_urls", "record_listings", "reset",
                    ],
                },
                "data": {"type": "object"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "generate_application",
        "description": "Izradjuje nacrt motivacijskog pisma za aktivan oglas i dostavljeni profil kandidata.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "listing_url": {"type": "string"},
                "applicant_profile": {"type": "string"},
            },
            "required": ["listing_url", "applicant_profile"],
        },
    },
]


def run_agent(user_query, mode="auto"):
    """Pokrece agenta i vraca odgovor za korisnika."""
    return run_task(user_query, mode=mode)["answer"]


def run_task(user_query, mode="auto"):
    """Izvrsava upit i vraca odgovor s podacima o izvrsavanju."""
    offline = _offline_mode()
    selected_mode = "heuristic" if offline else _select_mode(mode)
    executor = ToolExecutor(user_query, selected_mode)
    try:
        if offline:
            executor.log({
                "type": "offline_mode",
                "message": "Koriste se lokalni primjeri; Gemini i mrežni dohvat nisu dostupni.",
            })
        plan = _build_plan(user_query)
        # Pretrage s jasnim kriterijima moraju biti ponovljive. Gemini moze
        # parafrazirati kljucnu rijec i time promasiti sluzbeni portalni filter.
        # Zato se takvi upiti izvrsavaju deterministicki i kada je Gemini
        # dostupan; model ostaje za stvarno visekoracne razgovorne zadatke.
        if _heuristic_supported(plan):
            route = "heuristic" if selected_mode == "heuristic" else "deterministic_search"
            executor.log({"type": "plan", "route": route, "criteria": plan["criteria"]})
            answer, outcome = _execute_plan(plan, executor)
        elif selected_mode == "llm":
            executor.log({"type": "plan", "route": "llm_function_calling", "criteria": {}})
            answer, outcome = _run_llm_agent(user_query, executor, plan)
        else:
            executor.log({"type": "plan", "route": plan["route"], "criteria": plan["criteria"]})
            answer = (
                "Ovaj je upit visekoracni agentic zadatak (npr. usporedba, izracun, "
                "rad sa stanjem, provjera statusa ili motivacijsko pismo). Pokrenite ga "
                "s --mode llm i postavljenim GEMINI_API_KEY."
            )
            outcome = "needs_llm"
    except Exception as exc:
        answer = (
            "Zadatak nije izvrsen do kraja. Ne prikazujem nikakve neprovjerene "
            f"podatke. Greska: {exc}"
        )
        outcome = "error"
    finally:
        trace = executor.close(answer if "answer" in locals() else None, outcome if "outcome" in locals() else "error")
    return {"answer": answer, "outcome": outcome, "plan": trace["plan"], "trace": trace}


class ToolExecutor:
    """Poziva alate, biljezi izvrsavanje i ponavlja neuspjeli dohvat."""

    def __init__(self, task, mode):
        self.task = task
        self.mode = mode
        self.session_id = str(uuid.uuid4())[:8]
        self.step = 0
        self.events = []
        self.plan = None
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        self.log_file = open(LOG_PATH, "a", encoding="utf-8")
        self.log({"type": "task_start", "mode": mode, "model": MODEL if mode == "llm" else None, "max_steps": MAX_STEPS})

    def log(self, event):
        event = {"session_id": self.session_id, "task": self.task, **event}
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.events.append(event)
        self.log_file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self.log_file.flush()
        if event.get("type") == "plan":
            self.plan = event

    def call(self, tool_name, **arguments):
        if self.step >= MAX_STEPS:
            return {"error": f"Dosegnut je limit od {MAX_STEPS} koraka."}
        tool = {
            "search_jobs": search_jobs,
            "fetch_listing": fetch_listing,
            "estimate_earnings": estimate_earnings,
            "task_state": task_state,
            "generate_application": generate_application,
        }.get(tool_name)
        if tool is None:
            return {"error": f"Nepoznat alat: {tool_name}"}

        self.step += 1
        for attempt in range(1, MAX_RETRIES + 1):
            started = time.monotonic()
            self.log({"type": "tool_call", "step": self.step, "attempt": attempt, "tool": tool_name, "arguments": arguments})
            try:
                result = tool(**arguments)
            except Exception as exc:
                result = {"error": str(exc)}
            duration_ms = round((time.monotonic() - started) * 1000)
            failed = isinstance(result, dict) and "error" in result
            self.log({
                "type": "tool_result", "step": self.step, "attempt": attempt, "tool": tool_name,
                "duration_ms": duration_ms, "has_error": failed,
                "result_count": len(result) if isinstance(result, list) else None,
                "result_keys": sorted(result.keys()) if isinstance(result, dict) else None,
            })
            if not failed or attempt == MAX_RETRIES or tool_name in {"task_state", "estimate_earnings"}:
                return result
        return {"error": "Alat nije vratio rezultat."}

    def close(self, final_answer, outcome):
        self.log({"type": "task_end", "mode": self.mode, "total_steps": self.step, "outcome": outcome, "final_answer": (final_answer or "")[:500]})
        self.log_file.close()
        calls = [event for event in self.events if event["type"] == "tool_call"]
        return {"session_id": self.session_id, "total_steps": self.step, "tool_calls": calls, "plan": self.plan}


def _select_mode(mode):
    if mode not in {"auto", "llm", "heuristic"}:
        raise ValueError(f"Nepoznat mode: {mode}")
    if mode == "auto":
        return "llm" if os.environ.get("GEMINI_API_KEY") else "heuristic"
    if mode == "llm" and not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY nije postavljen. Koristi --mode heuristic ili postavi kljuc.")
    return mode


def _offline_mode():
    return os.environ.get("AGENT_OFFLINE") == "1"


def call_model_with_backoff(fn, *args, max_retries=3, base_wait=2, **kwargs):
    """Omotaj svaki poziv Gemini API-ju ovime, npr:
    response = call_model_with_backoff(model.generate_content, prompt)
    """
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            # Proširen uvjet da hvata i 429 i 503 greške
            error_str = str(e).upper()
            if any(err in error_str for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                if attempt == max_retries - 1:
                    break
                wait = base_wait * (2 ** attempt)
                print(f"[API opterecen] Cekam {wait}s prije ponovnog pokusaja ({attempt+1}/{max_retries}). Detalji: {e}")
                time.sleep(wait)
                continue
            raise  # neka druga greska - ne gutaj je neka pukne odmah
            
    raise RuntimeError("API se ne smiruje ni nakon nekoliko pokusaja.")


def _run_llm_agent(user_query, executor, plan=None):
    """Vodi Gemini kroz pozive alata i vraca zavrsni odgovor."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    declarations = [types.FunctionDeclaration(**item) for item in FUNCTION_DECLARATIONS]
    task_contract = _llm_task_contract(plan)
    history_only = bool(plan and plan.get("route") == "recent_listing_analysis")
    status_check = bool(plan and plan.get("route") == "listing_status")
    status_reference_title = plan.get("query", "") if status_check else ""
    config = types.GenerateContentConfig(
        system_instruction=LLM_SYSTEM_PROMPT + task_contract,
        tools=[types.Tool(function_declarations=declarations)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.1,
        max_output_tokens=1200,
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_query)])]
    search_attempts = 0
    empty_search_arguments = None
    recovery_requested = False
    candidate_urls = set()
    fetched_urls = set()
    verified_urls = set()
    required_history_urls = set()
    status_matched_urls = set()
    status_history_checked = False
    status_has_verified_page = False
    made_tool_call = False
    listing_fetch_calls = 0
    earnings_calls = 0

    last_model_call = None
    for turn in range(1, MAX_MODEL_TURNS + 1):
        if last_model_call is not None:
            # Ne usporavaj svaki uredan poziv unaprijed. Stvarno ogranicenje
            # API-ja obraduje call_model_with_backoff nakon odgovora 429/503.
            remaining = MIN_MODEL_CALL_INTERVAL_SECONDS - (time.monotonic() - last_model_call)
            if remaining > 0:
                time.sleep(remaining)
        last_model_call = time.monotonic()
        response = call_model_with_backoff(
            client.models.generate_content, model=MODEL, contents=contents, config=config
        )
        function_calls = response.function_calls or []
        if not function_calls:
            if empty_search_arguments and search_attempts < 2:
                fallback_arguments = _alternative_search_arguments(empty_search_arguments)
                executor.log({
                    "type": "recovery",
                    "reason": "prva pretraga nije vratila provjerene rezultate",
                    "suggested_arguments": fallback_arguments,
                })
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(
                            text="Prva pretraga nije vratila provjerene rezultate. Pokreni jos jednu "
                            "search_jobs pretragu s promijenjenim srodnim rijecima. Predlozeni argumenti su: "
                            + json.dumps(fallback_arguments, ensure_ascii=False)
                        )],
                    )
                )
                empty_search_arguments = None
                recovery_requested = True
                continue
            if recovery_requested:
                return "Agent nije pokrenuo alternativnu pretragu nakon praznog rezultata.", "error"
            if not made_tool_call:
                return "Gemini nije pozvao nijedan alat, pa nema provjerljivog odgovora.", "error"
            # Modelu ne namecemo provjeru svakog kandidata: to je prije
            # stvaralo vise krugova poziva iako je za odgovor dovoljno nekoliko
            # najboljih oglasa. Jedna provjera je najmanji dokaz za odgovor s
            # poveznicom; za usporedbu model sam bira do tri oglasa.
            pending_history_urls = required_history_urls - fetched_urls
            if pending_history_urls:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        text="Ovo je usporedba prethodno videnih oglasa. Prije zakljucka osvjezi "
                        "detalje svakog oglasa iz spremljenog skupa alatom fetch_listing: "
                        + ", ".join(sorted(pending_history_urls))
                    )],
                ))
                continue
            pending_status_urls = status_matched_urls - fetched_urls
            if pending_status_urls:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        text="U spremljenoj povijesti pronaden je trazeni oglas. Provjeri njegov izvorni URL "
                        "alatom fetch_listing prije zakljucka: " + ", ".join(sorted(pending_status_urls))
                    )],
                ))
                continue
            if status_check and status_history_checked and not status_matched_urls and search_attempts == 0:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        text="U spremljenoj povijesti nema podudarnog naslova. Sada smijes pokrenuti jednu "
                        "ciljanu search_jobs pretragu po naslovu i zatim provjeriti pronadeni URL."
                    )],
                ))
                continue
            pending_urls = candidate_urls - fetched_urls
            if pending_urls and not verified_urls and listing_fetch_calls < MAX_LLM_LISTING_FETCH_CALLS:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        text="Prije zavrsnog odgovora odaberi i provjeri jedan do tri najrelevantnija oglasa "
                        "alatom fetch_listing. Ne provjeravaj cijeli popis. Dostupni URL-ovi: "
                        + ", ".join(sorted(pending_urls)[:MAX_LLM_LISTING_FETCH_CALLS])
                    )],
                ))
                continue
            answer = (response.text or "").strip()
            if not answer:
                return "Nakon izvrsavanja alata nije dobiven zavrsni odgovor.", "error"
            if status_check and not status_has_verified_page:
                return (
                    "Nema provjerenog dokaza o trenutačnom statusu trazenog oglasa. "
                    "Ne zakljucujem da je neaktivan samo zato sto pretraga nije vratila rezultat.",
                    "no_verified_results",
                )
            executor.log({
                "type": "review",
                "search_attempts": search_attempts,
                "checked_listings": len(fetched_urls),
                "verified_listings": len(verified_urls),
            })
            citable_urls = verified_urls | (fetched_urls if status_check else set())
            return _validate_llm_answer(answer, citable_urls), "completed"

        model_content = response.candidates[0].content
        contents.append(model_content)
        response_parts = []
        for call in function_calls:
            arguments = dict(call.args or {})
            if history_only and call.name == "search_jobs":
                result = {"error": "Ovaj zadatak usporeduje spremljene oglase; nova pretraga nije dopustena."}
            elif status_check and call.name == "search_jobs" and not status_history_checked:
                result = {"error": "Najprije provjeri spremljene oglase alatom task_state."}
            elif status_check and call.name == "search_jobs" and status_matched_urls:
                result = {"error": "Podudarni spremljeni oglas postoji; provjeri njegov izvorni URL, ne pokreci novu pretragu."}
            elif call.name == "search_jobs" and search_attempts >= MAX_LLM_SEARCH_CALLS:
                result = {"error": "Dosegnut je limit od dvije pretrage; odaberi iz postojecih rezultata."}
            elif call.name == "fetch_listing" and listing_fetch_calls >= MAX_LLM_LISTING_FETCH_CALLS:
                result = {"error": "Dosegnut je limit od tri provjere oglasa; dovrsi odgovor s provjerenim podacima."}
            elif call.name == "estimate_earnings" and earnings_calls >= MAX_LLM_EARNINGS_CALLS:
                result = {"error": "Dosegnut je limit od tri izracuna; dovrsi usporedbu dostupnim rezultatima."}
            else:
                made_tool_call = True
                result = executor.call(call.name, **arguments)
            executor.log({
                "type": "observation",
                "turn": turn,
                "tool": call.name,
                "summary": _tool_observation(result),
            })
            if call.name == "search_jobs":
                if "error" not in result:
                    search_attempts += 1
                if search_attempts >= 2:
                    recovery_requested = False
                if isinstance(result, list):
                    candidate_urls.update(
                        url for item in result if isinstance(item, dict)
                        for url in [_listing_url(item)] if url
                    )
                if isinstance(result, list) and not result and search_attempts == 1:
                    empty_search_arguments = arguments
                else:
                    empty_search_arguments = None
            if call.name == "fetch_listing":
                if "error" not in result:
                    listing_fetch_calls += 1
                fetched_url = _listing_url(result) or arguments.get("url")
                if fetched_url:
                    fetched_urls.add(fetched_url)
                if arguments.get("url"):
                    fetched_urls.add(arguments["url"])
                if _valid_detail(result):
                    verified_url = _listing_url(result)
                    if verified_url:
                        verified_urls.add(verified_url)
                if status_check and isinstance(result, dict) and result.get("verified") is True:
                    status_has_verified_page = True
            if call.name == "estimate_earnings" and "error" not in result:
                earnings_calls += 1
            if call.name == "task_state" and arguments.get("action") == "get_recent_listings":
                if isinstance(result, dict):
                    if history_only:
                        required_history_urls.update(_urls_from_result(result.get("listings", [])))
                    if status_check:
                        status_history_checked = True
                        status_matched_urls.update(
                            _matching_listing_urls(result.get("listings", []), status_reference_title)
                        )
            if call.name == "generate_application" and isinstance(result, dict) and not result.get("error"):
                verified_urls.update(_urls_from_result(result))
            response_parts.append(types.Part.from_function_response(
                name=call.name,
                response={"result": _compact_tool_result(result)},
            ))
        contents.append(types.Content(role="user", parts=response_parts))

    return "Dosegnut je najveci broj koraka prije zavrsetka zadatka.", "error"


def _llm_task_contract(plan):
    """Daje modelu ogranicenje izvedeno iz namjere, ne iz pojedinog upita."""
    if not plan:
        return ""
    if plan.get("route") == "listing_status":
        return """

Ovo je provjera statusa jednog oglasa. Najprije obavezno pozovi task_state s
action=get_recent_listings i limit=20 te potrazi podudarni naslov u spremljenim
oglasima. Ako postoji, provjeri upravo njegov izvorni URL alatom fetch_listing;
nemoj pokretati novu pretragu. Samo ako nema podudarnog spremljenog oglasa,
napravi jednu ciljanu search_jobs pretragu po naslovu i provjeri rezultat.
Neaktivnost smijes tvrditi samo ako fetch_listing na konkretnoj stranici vrati
provjerenu negativnu aktivnost. Ako nema takvog dokaza, reci da aktivnost nije
moguce potvrditi — prazna pretraga nije dokaz neaktivnosti.
"""
    if plan.get("route") != "recent_listing_analysis":
        return ""
    return """

Ovaj je zadatak analiza oglasa koje je korisnik prethodno vidio, a ne nova
pretraga. Najprije obavezno pozovi task_state s action=get_recent_listings i
limit=3. Nemoj pozivati search_jobs. Za svaki vraceni URL pozovi fetch_listing
kako bi provjerio da je oglas jos aktivan i osvjezio podatke. Ako korisnik pita
za rok, usporedi samo potvrdene rokove. Ako pita za najveću zaradu ili odnos
satnice i radnog vremena, za svaki oglas koji ima nedvosmislenu satnicu i broj
sati tjedno pozovi estimate_earnings s weeks=1; nemoj pretpostavljati nedostajuce
sate ni razdoblje. Jasno reci kada se najveća zarada ne moze odrediti zbog
nedostajuceg podatka. Ne navodi URL s interpunkcijom na kraju poveznice.
"""


def _heuristic_supported(plan):
    """Heuristicki nacin zadrzava samo jednostavne filtrirane pretrage."""
    criteria = plan["criteria"]
    return (
        plan["route"] == "search"
        and not criteria.get("deadline_after")
        and not criteria.get("compare_newest")
        and not criteria.get("top_paid")
    )


def _alternative_search_arguments(arguments):
    query = _normalize(str(arguments.get("query") or ""))
    if _has_it_term(query) or "praksa" in query:
        alternative = "studentski programer razvoj softvera"
    elif "remote" in query or "online" in query or "daljinu" in query:
        alternative = "studentski posao rad od kuce"
    elif "vikend" in query:
        alternative = "studentski posao subota nedjelja"
    else:
        alternative = "studentski posao " + query[:80]
    return {
        "query": alternative,
        "city": arguments.get("city"),
        "min_wage": arguments.get("min_wage"),
    }


def _compact_tool_result(result):
    """Ogranicava velicinu opažanja koje se salje modelu."""
    if isinstance(result, list):
        return [
            {
                key: item.get(key)
                for key in (
                    "title", "url", "final_url", "snippet", "source_domain", "active",
                    "application_deadline", "detected_wage_eur_h", "posted_at",
                )
            }
            for item in result[:MAX_COMPACT_LISTINGS]
            if isinstance(item, dict)
        ]
    if isinstance(result, dict):
        compact = dict(result)
        if isinstance(compact.get("content_excerpt"), str):
            compact["content_excerpt"] = compact["content_excerpt"][:350]
        if isinstance(compact.get("letter"), str):
            compact["letter"] = compact["letter"][:2500]
        return compact
    return result


def _tool_observation(result):
    if isinstance(result, list):
        return {"result_count": len(result)}
    if isinstance(result, dict):
        return {"has_error": "error" in result, "fields": sorted(result.keys())[:12]}
    return {"result_type": type(result).__name__}


def _urls_from_result(result):
    if isinstance(result, list):
        return {url for item in result if isinstance(item, dict) for url in [_listing_url(item)] if url}
    if isinstance(result, dict):
        return {url for url in (_listing_url(result), result.get("listing_url")) if url}
    return set()


def _validate_llm_answer(answer, verified_urls):
    mentioned_urls = {
        _normalized_answer_url(url)
        for url in re.findall(r"https?://[^\s)]+", answer)
        if _normalized_answer_url(url)
    }
    verified_urls = {
        _normalized_answer_url(url)
        for url in verified_urls
        if _normalized_answer_url(url)
    }
    unknown_urls = mentioned_urls - verified_urls
    if unknown_urls:
        return (
            "Zavrsni odgovor sadrzavao je poveznicu koja nije dosla iz alata, "
            "zato nije prikazan kao provjeren rezultat."
        )
    return answer


def _normalized_answer_url(value):
    if not isinstance(value, str):
        return None
    return canonical_listing_url(value.rstrip(".,;:!?]}>\\\"'"))


def _build_plan(user_query, llm_hint=None):
    normalized = _normalize(user_query)
    city = _extract_city(user_query)
    criteria = {
        "city": city,
        "min_wage": _extract_min_wage(user_query),
        "min_wage_strict": _is_strict_min_wage(normalized),
        # "Bez prethodnog iskustva" je zasebna i cesta formulacija. Ovaj
        # kriterij se namjerno postavlja samo kada ga korisnik izricito trazi;
        # _matches_criteria zatim prihvaca samo oglase s pozitivnim dokazom.
        "no_experience": _has_any(normalized, (
            "bez iskustva", "bez radnog iskustva", "bez prethodnog iskustva",
            "bez prethodnog radnog iskustva", "ne traze iskustvo", "ne traže iskustvo",
            "nije potrebno iskustvo", "nije potrebno radno iskustvo",
            "nije potrebno prethodno iskustvo", "prethodno iskustvo nije potrebno",
            "prethodno radno iskustvo nije potrebno", "iskustvo nije potrebno",
            "iskustvo nije uvjet",
        )),
        "weekend_only": _has_any(normalized, ("iskljucivo vikendom", "vikendom", "subotom", "nedjeljom")),
        "remote": _has_any(normalized, ("remote", "rad od kuce", "rad od kuće", "na daljinu")),
        "internship": _has_any(normalized, ("praksa", "praksu", "prakse", "praksi", "internship", "trainee")),
        "recent_days": 3 if _has_any(normalized, ("zadnja 3 dana", "posljednja 3 dana")) else (7 if "ovaj tjedan" in normalized else None),
        "deadline_after": None,
        "category": _detect_category(normalized),
    }
    criteria["summer"] = _has_any(normalized, ("ljetn", "summer internship"))
    criteria["deadline_after"] = _extract_deadline_after(normalized)
    if _has_any(normalized, ("motivacijsko pismo", "prijavno pismo")):
        # Sprječava da zahtjev za pisanjem bez Gemini ključa bude pogrešno
        # tretiran kao obična pretraga oglasa. U LLM načinu puni upit dobiva
        # Gemini, koji sam odabire search/fetch/generate slijed.
        return {"route": "application", "criteria": criteria, "query": user_query}
    if _has_any(normalized, ("koliko", "zarad", "prag", "porez")) and _extract_hourly_wage(user_query):
        return {"route": "earnings", "criteria": criteria, "query": user_query}
    if _has_any(normalized, ("zapamti", "zapamti da", "zapamti samo")):
        return {"route": "save_preference", "criteria": criteria, "query": user_query}
    if _has_any(normalized, ("novi oglasi", "nove oglase", "od zadnji put", "sljedeci put")):
        return {"route": "new_listings", "criteria": criteria, "query": user_query}
    if _references_recent_listings(normalized):
        # Ova namjera je opca: korisnik se poziva na vlastitu povijest oglasa,
        # bez obzira usporeduje li rok, satnicu ili procijenjenu zaradu.
        return {"route": "recent_listing_analysis", "criteria": criteria, "query": user_query}
    if _has_any(normalized, (
        "jos aktivan", "još aktivan", "je li aktivan", "je li jos",
        "jos uvijek aktivan", "još uvijek aktivan", "provjeri je li oglas",
    )):
        return {"route": "listing_status", "criteria": criteria, "query": _extract_status_title(user_query)}

    criteria["compare_newest"] = _has_any(normalized, ("usporedi", "najnovij", "novije"))
    criteria["top_paid"] = _has_any(normalized, ("najbolje placen", "najplaćen", "najveca satnica"))
    criteria["limit"] = 3 if criteria["compare_newest"] or criteria["top_paid"] else 5
    return {"route": "search", "criteria": criteria, "query": _build_search_terms(user_query, llm_hint)}


def _execute_plan(plan, executor):
    if plan["route"] == "earnings":
        return _run_earnings(plan, executor)
    if plan["route"] == "save_preference":
        return _run_save_preference(plan, executor)
    if plan["route"] == "new_listings":
        return _run_new_listings(plan, executor)
    if plan["route"] == "recent_deadlines":
        return _run_recent_deadlines(executor)
    if plan["route"] == "listing_status":
        return _run_listing_status(plan, executor)
    return _run_search(plan, executor)


def _run_earnings(plan, executor):
    previous = executor.call("task_state", action="get_earnings")
    earned_so_far = previous.get("earned_so_far_eur", 0) if isinstance(previous, dict) else 0
    result = executor.call(
        "estimate_earnings", hourly_wage=_extract_hourly_wage(plan["query"]),
        hours_per_week=_extract_hours_per_week(plan["query"]),
        weeks=52 if _has_any(_normalize(plan["query"]), ("cijelu godinu", "cijele godine", "godinu")) else 1,
        earned_so_far=earned_so_far,
    )
    if "error" in result:
        return f"Izracun nije uspio: {result['error']}", "error"
    dependent = result["thresholds_eur"]["dependent_status"]
    tax = result["thresholds_eur"]["tax"]
    return "\n".join([
        f"Procijenjena zarada za trazeno razdoblje: {result['period_earnings_eur']:.2f} EUR.",
        f"Ukupno s prethodno zapamcenih {result['earned_so_far_eur']:.2f} EUR: {result['total_earnings_eur']:.2f} EUR.",
        f"Prag {dependent:.0f} EUR: {'dosegnut' if result['dependent_threshold_reached'] else 'nije dosegnut'}.",
        f"Prag {tax:.0f} EUR: {'dosegnut' if result['tax_threshold_reached'] else 'nije dosegnut'}.", result["assumptions"],
    ]), "completed"


def _run_save_preference(plan, executor):
    state = executor.call("task_state", action="set_preferences", data={"city": plan["criteria"]["city"], "keywords": _preferred_keywords(plan["query"])})
    if "error" in state:
        return f"Preferencije nisu spremljene: {state['error']}", "error"
    preference = state["preferences"]
    return f"Zapamtio sam preferenciju: {', '.join(preference['keywords']) or 'opci studentski poslovi'}; grad: {preference['city'] or 'bez odabranog grada'}.", "completed"


def _run_new_listings(plan, executor):
    preferences = executor.call("task_state", action="get_preferences")
    if "error" in preferences:
        return f"Ne mogu procitati prethodne preferencije: {preferences['error']}", "error"
    preference = preferences["preferences"]
    seen_state = executor.call("task_state", action="get_seen_urls")
    seen = set(seen_state.get("urls", [])) if isinstance(seen_state, dict) else set()
    terms = " ".join(preference.get("keywords") or []) or _build_search_terms(plan["query"], None)
    derived = {"route": "search", "query": terms, "criteria": {**plan["criteria"], "city": preference.get("city"), "limit": 5, "top_paid": False, "compare_newest": False}}
    listings, search_error = _find_verified_details(derived, executor)
    if search_error:
        return _tool_failure_text("pretraga", {"error": search_error}), "error"
    # Stanje sprema kanonski URL bez marketinških parametara. Tako ista
    # objava ne postaje "nova" samo zato što ju je portal ili tražilica vratio
    # s drukčijim UTM parametrom.
    new_listings = [
        item for item in listings
        if (canonical_listing_url(_listing_url(item)) or _listing_url(item)) not in seen
    ]
    if new_listings:
        executor.call("task_state", action="record_listings", data={"listings": new_listings})
    if not new_listings:
        return "Nema novih provjerenih oglasa od prethodnog prikaza za spremljene preferencije.", "no_verified_results"
    return _format_listings("Novi provjereni oglasi", new_listings, 5), "completed"


def _run_recent_deadlines(executor):
    saved = executor.call("task_state", action="get_recent_listings", data={"limit": 3})
    listings = saved.get("listings", []) if isinstance(saved, dict) else []
    if not listings:
        return "Nema zapamcena tri prethodno prikazana oglasa za usporedbu rokova.", "needs_history"
    refreshed = []
    for listing in listings:
        url = _listing_url(listing)
        if not url:
            continue
        detail = _get_listing_detail(listing, executor)
        if _valid_detail(detail) and detail.get("application_deadline"):
            refreshed.append({**listing, **detail})
    if not refreshed:
        return "Ni za jedan od posljednja tri oglasa nije potvrden aktivan oglas s vidljivim rokom prijave.", "no_verified_results"
    earliest = min(refreshed, key=lambda item: item["application_deadline"])
    return _format_listing("Najraniji provjereni rok prijave", earliest), "completed"


def _run_listing_status(plan, executor):
    candidates = executor.call("search_jobs", query=plan["query"], city=None, min_wage=None)
    if not isinstance(candidates, list):
        return _tool_failure_text("pretraga", candidates), "error"
    for candidate in candidates[:5]:
        detail = _get_listing_detail(candidate, executor)
        if _valid_detail(detail):
            return _format_listing("Oglas je trenutačno provjeren kao aktivan", {**candidate, **detail}), "completed"
    return "Nema provjerenog dokaza da je trazeni oglas trenutačno aktivan.", "no_verified_results"


def _run_search(plan, executor):
    listings, search_error = _find_verified_details(plan, executor)
    if search_error:
        return _tool_failure_text("pretraga", {"error": search_error}), "error"
    if not listings:
        return "Nema oglasa koji su prosli provjeru izvora, aktivnosti i trazenih uvjeta.", "no_verified_results"
    criteria = plan["criteria"]
    if criteria.get("compare_newest"):
        # Bez provjerljivog datuma objave ne smijemo tvrditi da je oglas
        # najnoviji. Takvi oglasi ostaju dostupni za običnu pretragu.
        listings = [item for item in listings if _parse_iso_date(item.get("posted_at"))]
        listings.sort(key=lambda item: item.get("posted_at") or "0000-00-00", reverse=True)
        title = "Najnoviji provjereni oglasi"
    elif criteria.get("top_paid"):
        listings = [item for item in listings if item.get("detected_wage_eur_h") is not None]
        listings.sort(key=lambda item: item["detected_wage_eur_h"], reverse=True)
        title = "Najbolje placeni provjereni oglasi"
    else:
        title = "Provjereni oglasi"
    listings = listings[:criteria.get("limit", 5)]
    if not listings:
        if criteria.get("compare_newest"):
            return "Nijedan aktivan provjereni oglas nema čitljiv datum objave potreban za poredak po novosti.", "no_verified_results"
        return "Nijedan aktivan provjereni oglas nema vidljivu satnicu potrebnu za usporedbu.", "no_verified_results"
    executor.call("task_state", action="record_listings", data={"listings": listings})
    answer = _format_listings(title, listings, len(listings))
    if criteria.get("compare_newest") and len(listings) < 3:
        answer += f"\nPronadena su samo {len(listings)} oglasa s dokazom trazenih uvjeta; ne dopunjavam popis nepovezanim oglasima."
    return answer, "completed"


def _find_verified_details(plan, executor):
    criteria = plan["criteria"]
    search_queries = ["studentski posao"] if _offline_mode() else _heuristic_search_queries(plan)
    last_error = None
    seen_urls = set()

    # Heuristika mora napraviti jedan srodan pokusaj kada prvi upit ne donese
    # provjeren pogodak. Ne trazimo izvan definiranih portala: ogranicenje je u
    # search_jobs, a grad i satnica ostaju isti u oba pokusaja.
    for attempt, search_query in enumerate(search_queries, 1):
        candidates = executor.call(
            "search_jobs", query=search_query, city=criteria.get("city"), min_wage=criteria.get("min_wage")
        )
        if not isinstance(candidates, list):
            last_error = candidates.get("error") if isinstance(candidates, dict) else "nepoznata greska pretrage"
            continue

        verified = []
        for candidate in candidates[:8]:
            url = _listing_url(candidate)
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            detail = _get_listing_detail(candidate, executor)
            if not _valid_detail(detail):
                continue
            item = {**candidate, **detail}
            if candidate.get("detected_wage_eur_h") is not None and item.get("detected_wage_eur_h") is None:
                item["detected_wage_eur_h"] = candidate["detected_wage_eur_h"]
            if candidate.get("posted_at") and not item.get("posted_at"):
                item["posted_at"] = candidate["posted_at"]
            if candidate.get("published") and not item.get("published"):
                item["published"] = candidate["published"]
            if _matches_criteria(item, criteria):
                verified.append(item)
        if verified:
            if attempt > 1:
                executor.log({"type": "recovery", "reason": "srodna heuristicka pretraga", "query": search_query})
            return verified, None

    return [], last_error


def _heuristic_search_queries(plan):
    """Vraca izvorni i najvise jedan ciljani srodni upit za pretragu oglasa."""
    primary = plan["query"]
    criteria = plan["criteria"]
    if criteria.get("category") == "it":
        alternative = "studentski IT posao praksa programer"
    elif criteria.get("category") == "hospitality":
        alternative = "studentski konobar sanker ugostiteljstvo"
    elif criteria.get("weekend_only"):
        alternative = "studentski posao vikendom subota nedjelja"
    elif criteria.get("remote"):
        alternative = "studentski posao remote rad od kuce"
    else:
        alternative = "studentski posao"
    return [primary] if _normalize(primary) == _normalize(alternative) else [primary, alternative]


def _matches_criteria(item, criteria):
    text = _normalize(" ".join(
        str(item.get(key) or "") for key in ("title", "content_excerpt", "location", "category")
    ))
    if criteria.get("no_experience") and item.get("experience_required") is not False:
        return False
    if criteria.get("weekend_only") and item.get("weekend_only") is not True:
        return False
    if criteria.get("remote") and item.get("remote") is not True:
        return False
    if criteria.get("internship") and not _is_internship_evidence(text):
        return False
    if criteria.get("summer") and not _has_any(text, ("ljetn", "summer internship")):
        return False
    if criteria.get("recent_days") and not _is_within_days(item.get("posted_at"), criteria["recent_days"]):
        return False
    if criteria.get("deadline_after"):
        deadline = _parse_iso_date(item.get("application_deadline"))
        if not deadline or deadline <= criteria["deadline_after"]:
            return False
    if criteria.get("city") and _normalize(criteria["city"]) not in text:
        return False
    if criteria.get("min_wage") is not None:
        wage = item.get("detected_wage_eur_h")
        if wage is None:
            return False
        if criteria.get("min_wage_strict") and wage <= criteria["min_wage"]:
            return False
        if not criteria.get("min_wage_strict") and wage < criteria["min_wage"]:
            return False
    category = criteria.get("category")
    if category and not _category_matches(category, text):
        return False
    return True


def _valid_detail(detail):
    return isinstance(detail, dict) and detail.get("verified") is True and detail.get("active") is True


def _get_listing_detail(listing, executor):
    """Vraca lokalni zapis u offline nacinu, a inace provjerava poveznicu."""
    if _offline_mode():
        return _offline_listing_detail(listing)
    detail = executor.call("fetch_listing", url=_listing_url(listing))
    # search_jobs je oglas vec potvrdio kao karticu na izravnom aktivnom
    # StudentPosao popisu. fetch_listing ne zna kontekst tog popisa pa zadrzi
    # taj dokaz samo kada pojedinacna stranica nema nikakav suprotan signal.
    if (
        isinstance(detail, dict)
        and listing.get("listed_in_active_feed")
        and detail.get("active") is False
        and detail.get("active_evidence") == "Nema pozitivnog dokaza da je oglas aktivan."
    ):
        detail = {
            **detail,
            "active": True,
            "active_evidence": "Oglas je upravo naveden na izravnom aktivnom popisu StudentPosao.hr.",
        }
    if (
        isinstance(detail, dict)
        and listing.get("listed_in_remote_feed")
        and detail.get("remote") is False
    ):
        detail = {
            **detail,
            "remote": True,
            "remote_evidence": "Oglas je upravo naveden u sluzbenom filtru Online posao na StudentPosao.hr.",
        }
    return detail


def _offline_listing_detail(listing):
    """Priprema podatke lokalnog primjera bez otvaranja mrežne poveznice."""
    title = str(listing.get("title") or "")
    snippet = str(listing.get("snippet") or "")
    text = _normalize(f"{title} {snippet}")
    url = _listing_url(listing)
    no_experience = _has_any(text, (
        "ne trazi iskustvo", "ne trazimo iskustvo", "nije potrebno iskustvo",
        "bez iskustva", "bez prethodnog iskustva", "bez radnog iskustva",
        "prethodno iskustvo nije potrebno", "iskustvo nije potrebno",
    ))
    weekend_only = _has_any(text, (
        "iskljucivo vikendom", "samo vikendom", "subota i nedjelja",
    ))
    remote = _has_any(text, ("rad na daljinu", "rad od kuce", "remote"))
    return {
        "verified": True,
        "active": True,
        "active_evidence": "Lokalni primjer iz sample_results.json (bez mrežne provjere).",
        "final_url": url,
        "source_domain": urlparse(url).netloc if url else None,
        "title": title or None,
        "content_excerpt": snippet,
        "detected_wage_eur_h": _extract_hourly_wage(text),
        "student_listing": "student" in text,
        "remote": remote,
        "remote_evidence": "Navod iz lokalnog primjera." if remote else None,
        "weekend_only": weekend_only,
        "weekend_evidence": "Navod iz lokalnog primjera." if weekend_only else None,
        "experience_required": False if no_experience else None,
        "experience_evidence": "U oglasu nije navedeno potrebno iskustvo." if no_experience else None,
    }


def _format_listings(heading, listings, limit):
    lines = [heading + ":"]
    for index, listing in enumerate(listings[:limit], 1):
        lines.append(_format_listing(f"{index}.", listing))
    if _offline_mode():
        lines.append("Prikazani su lokalni primjeri za offline rad; poveznice nisu otvorene preko mreze.")
    else:
        lines.append("Prikazani su samo oglasi s upravo provjerenim URL-om i dokazom aktivnosti.")
    return "\n".join(lines)


def _format_listing(label, listing):
    title = listing.get("title") or "Oglas bez naslova"
    lines = [f"{label} {title}", f"   Link: {_listing_url(listing) or 'nema dostupnog linka'}"]
    if listing.get("source_domain"):
        lines.append(f"   Izvor: {listing['source_domain']}")
    if listing.get("active_evidence"):
        lines.append(f"   Aktivnost: {listing['active_evidence']}")
    if listing.get("application_deadline"):
        lines.append(f"   Rok prijave: {listing['application_deadline']}")
    if listing.get("detected_wage_eur_h") is not None:
        lines.append(f"   Satnica: {listing['detected_wage_eur_h']:.2f} EUR/h")
    if listing.get("posted_at"):
        lines.append(f"   Objavljeno: {listing['posted_at']}")
    if listing.get("working_hours"):
        lines.append(f"   Radno vrijeme: {listing['working_hours']}")
    if listing.get("remote_evidence"):
        lines.append(f"   Remote dokaz: {listing['remote_evidence']}")
    if listing.get("weekend_evidence"):
        lines.append(f"   Vikend dokaz: {listing['weekend_evidence']}")
    if listing.get("experience_evidence"):
        lines.append(f"   Iskustvo: {listing['experience_evidence']}")
    return "\n".join(lines)


def _tool_failure_text(name, result):
    detail = result.get("error") if isinstance(result, dict) else "nepoznata greska"
    return f"Alat za {name} nije dao provjerljiv rezultat: {detail}"


def _build_search_terms(text, llm_hint):
    normalized = _normalize(text)
    if _has_it_term(normalized):
        return "studentska IT praksa programer developer"
    if _has_any(normalized, ("sanker", "šanker", "konobar", "ugostitelj")):
        return "studentski sanker konobar ugostiteljstvo"
    if _has_any(normalized, ("vikend", "subotom", "nedjeljom")):
        return "studentski vikend"
    if _has_any(normalized, ("remote", "rad od kuce", "rad od kuće", "na daljinu")):
        return "studentski remote"
    if llm_hint and llm_hint.get("route") == "search":
        suggested = _safe_search_terms(llm_hint.get("search_terms"))
        if suggested:
            return suggested
    words = [word for word in re.findall(r"[a-z0-9]+", normalized) if len(word) > 2 and word not in STOP_WORDS]
    return "studentski posao " + " ".join(words[:6]) if words else "studentski posao"


def _safe_search_terms(value):
    clean = " ".join(re.findall(r"[a-zA-Z0-9 ]+", value or "")).strip()
    return clean[:120] or None


def _preferred_keywords(text):
    normalized = _normalize(text)
    if _has_it_term(normalized):
        return ["IT"]
    terms = _build_search_terms(text, None)
    return [terms] if terms else []


def _extract_city(text):
    normalized = _normalize(text)
    for city in KNOWN_CITIES:
        if _normalize(city) in normalized:
            return city
    return None


def _extract_deadline_after(normalized_text):
    """Čita hrvatski datum nakon kojeg rok prijave mora biti kasniji."""
    month_numbers = {
        "sijecnja": 1, "veljace": 2, "ozujka": 3, "travnja": 4,
        "svibnja": 5, "lipnja": 6, "srpnja": 7, "kolovoza": 8,
        "rujna": 9, "listopada": 10, "studenoga": 11, "prosinca": 12,
    }
    match = re.search(
        r"(?:nakon|poslije|posle)\s+(\d{1,2})\.?\s*([a-z]+)(?:\s+(\d{4}))?",
        normalized_text,
    )
    if not match:
        return None
    day, month_name, year = match.groups()
    month = month_numbers.get(month_name)
    if not month:
        return None
    try:
        return date(int(year or date.today().year), month, int(day))
    except ValueError:
        return None


def _extract_status_title(text):
    quoted = _quoted_or_original(text)
    if quoted != text:
        return quoted
    match = re.search(
        r"oglas(?:\s+za)?\s+(.+?)(?:\s+jo[sš](?:\s+uvijek)?\s+aktivan\b|[?.!]?$)",
        text,
        re.I,
    )
    return match.group(1).strip().strip('"“”\'') if match else text


def _references_recent_listings(normalized_text):
    """Prepoznaje referencu na korisnikovu povijest, ne opceniti poredak oglasa."""
    has_listing_noun = bool(re.search(
        r"\b(?:oglas\w*|posao|posla|poslove|poslovi)\b",
        normalized_text,
    ))
    has_history_marker = _has_any(normalized_text, (
        "koja sam vidio", "koje sam vidio", "koje sam gledao", "koja sam gledao",
        "prethodno videne", "prethodno videni", "prethodno prikazane",
        "ranije videne", "ranije prikazane", "zadnje videne", "zadnja videna",
    )) or bool(re.search(
        r"\b(?:prethodn|ranij|zadnj)\w*\s+(?:vid|gled|prikaz)\w*\b",
        normalized_text,
    ))
    has_recent_quantity = bool(re.search(
        r"\b(?:zadnj(?:a|e|ih)|posljednj(?:a|e|ih)|prethodn(?:a|e|ih))\s+(?:\d+|tri|pet|nekoliko)\b",
        normalized_text,
    ))
    return has_listing_noun and (has_history_marker or (has_recent_quantity and _has_any(normalized_text, ("vidio", "videne", "gledao", "prikazane"))))


def _extract_min_wage(text):
    normalized = _normalize(text).replace(",", ".")
    for pattern in (r"(?:vise od|iznad|preko|minimalno|najmanje|>=|>)\s*(\d+(?:\.\d+)?)\s*(?:eur|€)?", r"(\d+(?:\.\d+)?)\s*(?:eur|€)\s*/?\s*(?:h|sat)"):
        match = re.search(pattern, normalized)
        if match:
            return float(match.group(1))
    return None


def _is_strict_min_wage(normalized_text):
    """Razlikuje 'vise od 9' od 'najmanje 9' i operatora >=."""
    if _has_any(normalized_text, ("vise od", "iznad", "preko")):
        return True
    return bool(re.search(r"(?<![<>=])>(?!=)\s*\d", normalized_text))


def _extract_hourly_wage(text):
    normalized = _normalize(text).replace(",", ".")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:eur|€)\s*/?\s*(?:h|sat)", normalized)
    return float(match.group(1)) if match else None


def _extract_hours_per_week(text):
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*sati?(?:\s+tjedno)?", _normalize(text))
    if not match:
        raise ValueError("U upitu nije naveden broj sati rada tjedno.")
    return float(match.group(1).replace(",", "."))


def _quoted_or_original(text):
    match = re.search(r'["“\'](.+?)["”\']', text)
    return match.group(1) if match else text


def _listing_url(listing):
    return (listing.get("final_url") or listing.get("url")) if isinstance(listing, dict) else None


def _matching_listing_urls(listings, requested_title):
    """Vraca spremljene URL-ove ciji naslov dovoljno pouzdano opisuje upit."""
    requested = _normalize(requested_title)
    requested_words = _title_words(requested)
    if not requested or not requested_words:
        return set()

    matches = set()
    for listing in listings if isinstance(listings, list) else []:
        if not isinstance(listing, dict):
            continue
        title = _normalize(str(listing.get("title") or ""))
        url = _listing_url(listing)
        if not title or not url:
            continue
        if title == requested or title in requested or requested in title:
            matches.add(url)
            continue
        title_words = _title_words(title)
        overlap = len(requested_words & title_words)
        # Izbjegava podudaranje po samo jednoj opcenitoj rijeci poput "posao",
        # ali prihvaca naslov kojemu su izostavljeni nebitni interpunkcijski dijelovi.
        if overlap >= 2 and overlap / min(len(requested_words), len(title_words)) >= 0.8:
            matches.add(url)
    return matches


def _title_words(value):
    return {
        word for word in re.findall(r"[a-z0-9]+", value)
        if len(word) > 2 and word not in STOP_WORDS
    }


def _parse_iso_date(value):
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def _is_within_days(value, days):
    parsed = _parse_iso_date(value)
    return bool(parsed and date.today() - timedelta(days=days) <= parsed <= date.today())


def _has_any(text, phrases):
    return any(_normalize(phrase) in text for phrase in phrases)


def _detect_category(text):
    if _has_any(text, ("sanker", "šanker")):
        return "bartender"
    if _has_any(text, ("konobar", "ugostitelj", "fast food", "kuhar")):
        return "hospitality"
    if _has_it_term(text):
        return "it"
    return None


def _has_it_term(text):
    """Prepoznaje IT kao rijec, ne kao dio rijeci poput 'raditi'."""
    return bool(re.search(r"\bit\b", text)) or _has_any(
        text, ("programer", "developer", "informat", "softver", "software")
    )


def _is_internship_evidence(text):
    return _has_any(text, ("praksa", "praksi", "praktik", "internship", "trainee"))


def _category_matches(category, text):
    markers = {
        "bartender": ("sanker", "šanker"),
        "hospitality": ("konobar", "sanker", "šanker", "kuhar", "fast food", "ugostitelj"),
        "it": (" it ", "programer", "developer", "informat", "softver", "software"),
    }
    return _has_any(" " + text + " ", markers[category])


def _normalize(text):
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in text if not unicodedata.combining(char))


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Agent za pronalazak studentskih poslova.")
    # Naredbena ljuska uklanja unutarnje navodnike. Prihvaćanjem svih
    # preostalih dijelova upita korisnik može provjeriti naslov u navodnicima
    # bez argparse pogreške.
    parser.add_argument("query", nargs="+", help="Korisnicki upit")
    parser.add_argument("--mode", choices=["auto", "llm", "heuristic"], default=os.environ.get("AGENT_MODE", "auto"))
    parser.add_argument("--offline", action="store_true", help="Koristi lokalne primjere umjesto live pretrage.")
    args = parser.parse_args(argv)
    args.query = " ".join(args.query)
    return args


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parse_args(sys.argv[1:])
    if args.offline:
        os.environ["AGENT_OFFLINE"] = "1"
    print("\n--- ODGOVOR ---")
    print(run_agent(args.query, mode=args.mode))
