import os
import sys
import tempfile
import unittest
from datetime import date, timedelta
from types import ModuleType
from unittest.mock import patch

from agent import (
    _alternative_search_arguments,
    _build_plan,
    call_model_with_backoff,
    _category_matches,
    _find_verified_details,
    _get_listing_detail,
    _heuristic_search_queries,
    _heuristic_supported,
    _matches_criteria,
    _parse_args,
    _run_search,
    _is_strict_min_wage,
    _has_it_term,
    _run_llm_agent,
    run_task,
)
from estimate_earnings import estimate_earnings
from fetch_listing import _experience_required, _weekend_only
from generate_application import _validate_profile, generate_application
from search_jobs import _parse_published_date
from task_state import task_state


class AgentFeatureTests(unittest.TestCase):
    def test_annual_earnings(self):
        result = estimate_earnings(7, 20, weeks=52)
        self.assertEqual(result["period_earnings_eur"], 7280.0)
        self.assertTrue(result["dependent_threshold_reached"])
        self.assertFalse(result["tax_threshold_reached"])

    def test_state_persists_preferences_and_recent_listings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.json")
            task_state("set_preferences", {"city": "Osijek", "keywords": ["IT"]}, path)
            task_state("record_listings", {"listings": [{
                "title": "Testni oglas", "final_url": "https://studentposao.hr/oglas/test/1",
                "verified": True, "active": True, "application_deadline": "2099-12-31",
            }]}, path)
            self.assertEqual(task_state("get_preferences", state_path=path)["preferences"]["city"], "Osijek")
            self.assertEqual(len(task_state("get_recent_listings", {"limit": 3}, path)["listings"]), 1)

    def test_state_deduplicates_listing_tracking_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.json")
            task_state("record_listings", {"listings": [{
                "title": "Prvi prikaz", "final_url": "https://studentposao.hr/oglas/test/1?utm_source=portal",
            }]}, path)
            task_state("record_listings", {"listings": [{
                "title": "Ponovljeni prikaz", "final_url": "https://studentposao.hr/oglas/test/1?fbclid=abc",
            }]}, path)
            seen = task_state("get_seen_urls", state_path=path)["urls"]
            self.assertEqual(seen, ["https://studentposao.hr/oglas/test/1"])

    def test_corrupt_state_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.json")
            with open(path, "w", encoding="utf-8") as state_file:
                state_file.write("nije json")
            with self.assertRaises(ValueError):
                task_state("get_seen_urls", state_path=path)
            with open(path, encoding="utf-8") as state_file:
                self.assertEqual(state_file.read(), "nije json")

    def test_newest_search_rejects_listing_without_date(self):
        class Executor:
            def call(self, tool_name, **arguments):
                self.assertEqual(tool_name, "task_state")
                return {"recorded": 0}

            def assertEqual(self, left, right):
                if left != right:
                    raise AssertionError(f"{left} != {right}")

        plan = {"criteria": {"compare_newest": True, "limit": 3}}
        with patch("agent._find_verified_details", return_value=([{
            "title": "Bez datuma", "url": "https://studentposao.hr/oglas/test/1",
        }], None)):
            answer, outcome = _run_search(plan, Executor())
        self.assertEqual(outcome, "no_verified_results")
        self.assertIn("datum objave", answer)

    def test_experience_is_unknown_without_explicit_evidence(self):
        self.assertIsNone(_experience_required("opis oglasa bez uvjeta"))
        self.assertFalse(_experience_required("nije potrebno iskustvo"))
        self.assertTrue(_experience_required("potrebno je radno iskustvo"))

    def test_extended_no_experience_phrases(self):
        self.assertFalse(_experience_required("Nije potrebno radno iskustvo."))
        self.assertFalse(_experience_required("Prethodno iskustvo nije potrebno."))
        self.assertFalse(_experience_required("Posao je dostupan bez prethodnog iskustva."))
        self.assertFalse(_experience_required("Nije potrebno prethodno radno iskustvo."))

    def test_no_experience_query_variants_enable_strict_filter(self):
        for query in (
            "Pronadi studentske poslove bez prethodnog iskustva",
            "Pronadi studentske poslove bez prethodnog radnog iskustva",
            "Pronadi studentske poslove gdje prethodno iskustvo nije potrebno",
        ):
            self.assertTrue(_build_plan(query)["criteria"]["no_experience"], query)
        criteria = _build_plan("Pronadi studentske poslove bez prethodnog iskustva")["criteria"]
        self.assertTrue(_matches_criteria({"experience_required": False}, criteria))
        self.assertFalse(_matches_criteria({"experience_required": True}, criteria))
        self.assertFalse(_matches_criteria({"experience_required": None}, criteria))

    def test_weekend_schedule_requires_no_weekday_evidence(self):
        self.assertTrue(_weekend_only("Rad vikendom: subota i nedjelja."))
        self.assertFalse(_weekend_only("Rad vikendom i radnim danima po dogovoru."))
        self.assertFalse(_weekend_only("Spremnost na rad vikendom je prednost."))
        self.assertFalse(_weekend_only(
            "Ne radimo subotom, nedjeljom i praznikom, stoga posao nikada nece omesti tvoje vikend planove."
        ))

    def test_more_than_wage_excludes_equal_wage(self):
        item = {"detected_wage_eur_h": 9}
        self.assertFalse(_matches_criteria(item, {"min_wage": 9, "min_wage_strict": True}))
        self.assertTrue(_matches_criteria({"detected_wage_eur_h": 9.01}, {"min_wage": 9, "min_wage_strict": True}))
        self.assertFalse(_is_strict_min_wage("najmanje 9 eur/h"))
        self.assertFalse(_is_strict_min_wage(">= 9 eur/h"))
        self.assertTrue(_is_strict_min_wage("> 9 eur/h"))

    def test_it_internship_requires_both_it_category_and_practice_evidence(self):
        plan = _build_plan("Pronadi studentsku praksu u IT djelatnostima")
        self.assertTrue(plan["criteria"]["internship"])
        self.assertEqual(plan["criteria"]["category"], "it")
        self.assertTrue(_matches_criteria({
            "category": "it i digitalno", "content_excerpt": "Placena strucna praksa.",
        }, plan["criteria"]))
        self.assertFalse(_matches_criteria({
            "category": "it i digitalno", "content_excerpt": "Stalni posao programera.",
        }, plan["criteria"]))

    def test_summer_internship_requires_summer_and_practice_evidence(self):
        plan = _build_plan("Ima li ljetnih praksi za informatiku u Zagrebu s rokom prijave poslije 15. rujna?")
        self.assertTrue(plan["criteria"]["internship"])
        self.assertTrue(plan["criteria"]["summer"])
        self.assertEqual(plan["criteria"]["deadline_after"], date(date.today().year, 9, 15))
        self.assertFalse(_matches_criteria({
            "category": "it i digitalno", "content_excerpt": "Junior Software Developer.",
            "application_deadline": "2099-12-31",
        }, plan["criteria"]))
        self.assertTrue(_matches_criteria({
            "category": "it i digitalno", "content_excerpt": "Ljetna stručna praksa za studente informatike u Zagrebu.",
            "application_deadline": "2099-12-31",
        }, plan["criteria"]))

    def test_application_request_without_gemini_requires_agentic_mode_and_does_not_list_jobs(self):
        query = (
            "Napiši mi motivacijsko pismo za posao Junior Software Developer. "
            "Profil: Student 3. godine prijediplomskog studija matematike i računarstva, "
            "imam iskustva s raznim programskim jezicima."
        )
        with tempfile.TemporaryDirectory() as directory, \
             patch("agent.LOG_PATH", os.path.join(directory, "execution_log.jsonl")), \
             patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False), \
             patch("agent.search_jobs") as search, \
             patch("agent.generate_application") as application:
            result = run_task(query, mode="heuristic")

        self.assertEqual(result["outcome"], "needs_llm")
        self.assertFalse(search.called)
        self.assertFalse(application.called)
        self.assertIn("GEMINI_API_KEY", result["answer"])
        self.assertNotIn("Provjereni oglasi:", result["answer"])

    def test_recent_deadlines_with_word_tri_is_not_search(self):
        plan = _build_plan("Od zadnja tri oglasa koje sam gledao, koji ima najraniji rok prijave?")
        self.assertEqual(plan["route"], "recent_listing_analysis")

    def test_recent_listing_reference_covers_deadline_and_earnings_questions(self):
        for query in (
            "Od zadnja 3 oglasa koja sam vidio, koji posao ce platiti najvise?",
            "Koji od prethodno prikazana tri posla ima najraniji rok prijave?",
            "Od posljednja tri oglasa koje sam gledao, koji nudi najvecu zaradu?",
        ):
            self.assertEqual(_build_plan(query)["route"], "recent_listing_analysis", query)

    def test_status_with_quoted_title_is_not_search(self):
        plan = _build_plan('Provjeri je li oglas za "Junior Software Developer" još uvijek aktivan.')
        self.assertEqual(plan["route"], "listing_status")
        self.assertEqual(plan["query"], "Junior Software Developer")

    def test_status_with_single_quoted_title_uses_only_the_listing_title(self):
        plan = _build_plan(
            "Provjeri je li oglas 'Nosenje maskota, animacija djece - DALAL OBRT' jos aktivan."
        )
        self.assertEqual(plan["route"], "listing_status")
        self.assertEqual(plan["query"], "Nosenje maskota, animacija djece - DALAL OBRT")

    def test_status_without_quotes_extracts_only_listing_title(self):
        plan = _build_plan("Provjeri je li oglas za Junior Software Developer još uvijek aktivan.")
        self.assertEqual(plan["route"], "listing_status")
        self.assertEqual(plan["query"], "Junior Software Developer")

    def test_cli_accepts_shell_split_inner_quotes(self):
        args = _parse_args([
            "Provjeri je li oglas za Junior", "Software", "Developer još uvijek aktivan.",
        ])
        self.assertEqual(args.query, "Provjeri je li oglas za Junior Software Developer još uvijek aktivan.")

    def test_application_without_gemini_key_does_not_fetch_or_generate_template(self):
        url = "https://studentposao.hr/oglas/test/1"
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False), \
             patch("generate_application.fetch_listing") as fetch:
            result = generate_application(url, "Student sam računarstva i radim s Pythonom, Javom i bazama podataka.")
        self.assertIn("error", result)
        self.assertIn("GEMINI_API_KEY", result["error"])
        self.assertFalse(fetch.called)

    def test_complex_contract_routes_require_llm_without_key(self):
        for query in (
            "Ima li otvorenih ljetnih IT praksi u Zagrebu s rokom prijave nakon 1. rujna?",
            "Usporedi tri najnovija ugostiteljska posla u Osijeku po satnici.",
            "Ako radim cijelu godinu 20 sati tjedno za 7 EUR/h, kolika je zarada?",
            "Daj tri najbolje placena oglasa za sankera u Osijeku ovaj tjedan.",
        ):
            with tempfile.TemporaryDirectory() as directory, \
                 patch("agent.LOG_PATH", os.path.join(directory, "execution_log.jsonl")), \
                 patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
                result = run_task(query, mode="heuristic")
            self.assertEqual(result["outcome"], "needs_llm", query)

    def test_online_query_is_not_misclassified_as_it(self):
        plan = _build_plan("Pronadi sve poslove koje mogu raditi online ili remote")
        self.assertTrue(plan["criteria"]["remote"])
        self.assertIsNone(plan["criteria"]["category"])
        self.assertFalse(_has_it_term("mogu raditi online"))
        self.assertTrue(_has_it_term("IT programer"))

    def test_llm_mode_uses_deterministic_path_for_simple_search(self):
        def local_search(query, city=None, min_wage=None):
            return [{
                "title": "Test oglas", "url": "https://studentposao.hr/oglas/test/1",
                "verified": True, "active": True, "detected_wage_eur_h": 10,
            }]

        def local_detail(url):
            return {
                "verified": True, "active": True, "final_url": url,
                "detected_wage_eur_h": 10, "content_excerpt": "Osijek",
            }

        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False), \
             patch("agent.LOG_PATH", os.path.join(directory, "execution_log.jsonl")), \
             patch("agent.search_jobs", side_effect=local_search), \
             patch("agent.fetch_listing", side_effect=local_detail), \
             patch("agent.task_state", return_value={"recorded": 1}), \
             patch("agent._run_llm_agent") as llm_agent:
            result = run_task("Pronadi studentske poslove u Osijeku koji placaju vise od 9 EUR/h", mode="llm")

        self.assertEqual(result["outcome"], "completed")
        self.assertFalse(llm_agent.called)
        self.assertIn("Test oglas", result["answer"])

    def test_remote_feed_evidence_survives_detail_fetch(self):
        class Executor:
            def call(self, tool_name, **arguments):
                self.assertEqual(tool_name, "fetch_listing")
                return {"verified": True, "active": True, "remote": False}

            def assertEqual(self, left, right):
                if left != right:
                    raise AssertionError(f"{left} != {right}")

        detail = _get_listing_detail(
            {"url": "https://studentposao.hr/oglas/test/1", "listed_in_remote_feed": True},
            Executor(),
        )
        self.assertTrue(detail["remote"])
        self.assertIn("Online posao", detail["remote_evidence"])

    def test_relative_published_hour_is_today(self):
        self.assertEqual(_parse_published_date("prije 4 h"), date.today().isoformat())
        self.assertEqual(_parse_published_date("prije 2 sata"), date.today().isoformat())
        self.assertEqual(_parse_published_date("prije 5 sati"), date.today().isoformat())
        self.assertEqual(_parse_published_date("prije 20 min"), date.today().isoformat())

    def test_relative_published_day_forms(self):
        self.assertEqual(
            _parse_published_date("prije 1 dan"),
            (date.today() - timedelta(days=1)).isoformat(),
        )
        self.assertEqual(
            _parse_published_date("prije 3 dana"),
            (date.today() - timedelta(days=3)).isoformat(),
        )
        self.assertEqual(
            _parse_published_date("jučer"),
            (date.today() - timedelta(days=1)).isoformat(),
        )

    def test_hyphenated_published_date_is_parsed(self):
        self.assertEqual(_parse_published_date("14-09-2026"), "2026-09-14")

    def test_hospitality_filter_does_not_accept_unrelated_listing(self):
        self.assertTrue(_category_matches("hospitality", "konobar u kaficu"))
        self.assertFalse(_category_matches("hospitality", "pomocni radnik u skladistu"))

    def test_heuristic_mode_is_limited_to_simple_searches(self):
        simple_plan = _build_plan("Pronadi studentske poslove u Osijeku bez iskustva")
        complex_plan = _build_plan("Ako radim 20 sati tjedno za 7 EUR/h, kolika je zarada?")
        self.assertTrue(_heuristic_supported(simple_plan))
        self.assertFalse(_heuristic_supported(complex_plan))

    def test_offline_mode_skips_gemini_and_listing_http_requests(self):
        def local_state(action, data=None):
            if action == "record_listings":
                return {"recorded": len((data or {}).get("listings", []))}
            return {"preferences": {"city": None, "keywords": []}, "urls": [], "listings": []}

        with tempfile.TemporaryDirectory() as directory:
            log_path = os.path.join(directory, "execution_log.jsonl")
            with patch.dict(os.environ, {"AGENT_OFFLINE": "1", "GEMINI_API_KEY": "test-key"}, clear=False), \
                 patch("agent.LOG_PATH", log_path), \
                 patch("agent.task_state", side_effect=local_state), \
                 patch("agent._run_llm_agent") as llm_agent, \
                 patch("agent.fetch_listing", side_effect=AssertionError("Mrezni dohvat nije dopusten u offline nacinu")):
                started = __import__("time").monotonic()
                result = run_task(
                    "Pronadi studentske poslove u Osijeku koji placaju vise od 8 EUR/h i ne traze iskustvo.",
                    mode="llm",
                )

        self.assertLess(__import__("time").monotonic() - started, 2)
        self.assertEqual(result["outcome"], "completed")
        self.assertIn("Konobar/ica", result["answer"])
        self.assertFalse(llm_agent.called)
        self.assertNotIn("fetch_listing", [call["tool"] for call in result["trace"]["tool_calls"]])

    def test_alternative_search_keeps_city_and_wage(self):
        alternative = _alternative_search_arguments({"query": "IT praksa", "city": "Osijek", "min_wage": 8})
        self.assertEqual(alternative["query"], "studentski programer razvoj softvera")
        self.assertEqual(alternative["city"], "Osijek")
        self.assertEqual(alternative["min_wage"], 8)

    def test_alternative_search_recovers_online_query(self):
        alternative = _alternative_search_arguments({"query": "studentski posao online", "city": None, "min_wage": None})
        self.assertEqual(alternative["query"], "studentski posao rad od kuce")

    def test_heuristic_adds_one_targeted_fallback_query(self):
        plan = _build_plan("Pronadi studentske poslove koji se rade iskljucivo vikendom")
        self.assertEqual(
            _heuristic_search_queries(plan),
            ["studentski vikend", "studentski posao vikendom subota nedjelja"],
        )

    def test_heuristic_uses_fallback_after_empty_search(self):
        plan = _build_plan("Pronadi studentske poslove koji se rade iskljucivo vikendom")
        executor = _SearchExecutor([[], [{
            "title": "Konobar vikendom", "url": "https://studentposao.hr/oglas/test/1",
        }]])
        listings, error = _find_verified_details(plan, executor)

        self.assertIsNone(error)
        self.assertEqual(len(listings), 1)
        self.assertEqual(len(executor.search_queries), 2)
        self.assertIn("subota", executor.search_queries[1])

    def test_gemini_backoff_does_not_sleep_after_final_failed_attempt(self):
        def unavailable():
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

        with patch("agent.time.sleep") as sleep:
            with self.assertRaises(RuntimeError):
                call_model_with_backoff(unavailable, max_retries=3, base_wait=2)

        self.assertEqual(sleep.call_args_list, [((2,),), ((4,),)])

    def test_application_profile_must_not_be_empty(self):
        with self.assertRaises(ValueError):
            _validate_profile("premalo")

    def test_llm_agent_uses_search_then_fetch_before_answer(self):
        executor = _FakeExecutor()
        responses = [
            _FakeResponse([_FakeFunctionCall("search_jobs", {"query": "IT", "city": "Osijek"})]),
            _FakeResponse([_FakeFunctionCall("fetch_listing", {"url": "https://studentposao.hr/oglas/test/1"})]),
            _FakeResponse([], "Pronaden je oglas: https://studentposao.hr/oglas/test/1"),
        ]
        google_module, genai_module = _fake_google_modules(responses)
        with patch.dict(sys.modules, {"google": google_module, "google.genai": genai_module}), patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False
        ):
            answer, outcome = _run_llm_agent("Pronadi IT posao", executor)

        self.assertEqual(outcome, "completed")
        self.assertIn("https://studentposao.hr/oglas/test/1", answer)
        self.assertEqual([call[0] for call in executor.calls], ["search_jobs", "fetch_listing"])

    def test_llm_agent_does_not_force_detail_fetch_for_every_search_result(self):
        candidates = [
            {
                "title": f"IT praksa {index}",
                "final_url": f"https://studentposao.hr/oglas/test/{index}",
            }
            for index in range(1, 5)
        ]
        executor = _FakeExecutor(search_results=[candidates])
        responses = [
            _FakeResponse([_FakeFunctionCall("search_jobs", {"query": "IT praksa", "city": "Osijek"})]),
            # Model pokusava zavrsiti prerano. Agent mora zatraziti samo
            # odabranu provjeru, a ne otvoriti sva cetiri rezultata.
            _FakeResponse([], "Pronadeni su oglasi."),
            _FakeResponse([_FakeFunctionCall("fetch_listing", {"url": "https://studentposao.hr/oglas/test/1"})]),
            _FakeResponse([], "Pronadena je praksa: https://studentposao.hr/oglas/test/1"),
        ]
        google_module, genai_module = _fake_google_modules(responses)
        with patch.dict(sys.modules, {"google": google_module, "google.genai": genai_module}), patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False
        ):
            answer, outcome = _run_llm_agent("Pronadi IT praksu", executor)

        self.assertEqual(outcome, "completed")
        self.assertIn("https://studentposao.hr/oglas/test/1", answer)
        self.assertEqual([call[0] for call in executor.calls], ["search_jobs", "fetch_listing"])

    def test_llm_agent_requests_alternative_search_after_empty_result(self):
        executor = _FakeExecutor(search_results=[[], []])
        responses = [
            _FakeResponse([_FakeFunctionCall("search_jobs", {"query": "IT praksa", "city": "Osijek"})]),
            _FakeResponse([], "Nema rezultata."),
            _FakeResponse([_FakeFunctionCall("search_jobs", {"query": "programer", "city": "Osijek"})]),
            _FakeResponse([], "Nema provjerenih oglasa za trazeni upit."),
        ]
        google_module, genai_module = _fake_google_modules(responses)
        with patch.dict(sys.modules, {"google": google_module, "google.genai": genai_module}), patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False
        ):
            answer, outcome = _run_llm_agent("Pronadi IT praksu", executor)

        self.assertEqual(outcome, "completed")
        self.assertIn("Nema provjerenih oglasa", answer)
        self.assertEqual([call[0] for call in executor.calls], ["search_jobs", "search_jobs"])
        self.assertTrue(any(event["type"] == "recovery" for event in executor.events))

    def test_llm_history_analysis_uses_state_not_a_new_search_and_refreshes_all_listings(self):
        query = "Od zadnja 3 oglasa koja sam vidio, koji posao ce platiti najvise i koliko?"
        plan = _build_plan(query)
        executor = _HistoryExecutor()
        urls = [listing["final_url"] for listing in executor.listings]
        responses = [
            _FakeResponse([_FakeFunctionCall("task_state", {"action": "get_recent_listings", "data": {"limit": 3}})]),
            # Nova pretraga je za upit o povijesti neprihvatljiva; agent je
            # odbija i model nastavlja s podacima iz task_state.
            _FakeResponse([_FakeFunctionCall("search_jobs", {"query": "studentski posao"})]),
            _FakeResponse([_FakeFunctionCall("fetch_listing", {"url": url}) for url in urls]),
            _FakeResponse([
                _FakeFunctionCall("estimate_earnings", {"hourly_wage": 8 + index, "hours_per_week": 10, "weeks": 1})
                for index in range(3)
            ]),
            _FakeResponse([], f"Najveca tjedna zarada je 100 EUR: {urls[2]} ."),
        ]
        google_module, genai_module = _fake_google_modules(responses)
        with patch.dict(sys.modules, {"google": google_module, "google.genai": genai_module}), patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False
        ):
            answer, outcome = _run_llm_agent(query, executor, plan)

        self.assertEqual(outcome, "completed")
        self.assertIn("Najveca tjedna zarada", answer)
        self.assertNotIn("Zavrsni odgovor sadrzavao", answer)
        self.assertNotIn("search_jobs", [call[0] for call in executor.calls])
        self.assertEqual([call[0] for call in executor.calls].count("fetch_listing"), 3)
        self.assertEqual([call[0] for call in executor.calls].count("estimate_earnings"), 3)

    def test_llm_status_checks_matching_saved_url_before_search(self):
        query = "Provjeri je li oglas 'Nosenje maskota, animacija djece - DALAL OBRT' jos aktivan."
        plan = _build_plan(query)
        executor = _StatusExecutor()
        url = executor.listings[0]["final_url"]
        responses = [
            _FakeResponse([_FakeFunctionCall("task_state", {"action": "get_recent_listings", "data": {"limit": 20}})]),
            # Pokusaj nove pretrage mora biti odbijen jer spremljeni naslov vec
            # ima izvorni URL koji je potrebno provjeriti.
            _FakeResponse([_FakeFunctionCall("search_jobs", {"query": "Nosenje maskota"})]),
            _FakeResponse([_FakeFunctionCall("fetch_listing", {"url": url})]),
            _FakeResponse([], f"Oglas je aktivan: {url}."),
        ]
        google_module, genai_module = _fake_google_modules(responses)
        with patch.dict(sys.modules, {"google": google_module, "google.genai": genai_module}), patch.dict(
            os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False
        ):
            answer, outcome = _run_llm_agent(query, executor, plan)

        self.assertEqual(outcome, "completed")
        self.assertIn("Oglas je aktivan", answer)
        self.assertNotIn("search_jobs", [call[0] for call in executor.calls])
        self.assertEqual([call[0] for call in executor.calls].count("fetch_listing"), 1)


class _FakeExecutor:
    def __init__(self, search_results=None):
        self.calls = []
        self.events = []
        self.search_results = list(search_results or [[{
            "title": "IT praksa", "final_url": "https://studentposao.hr/oglas/test/1",
        }]])

    def call(self, tool_name, **arguments):
        self.calls.append((tool_name, arguments))
        if tool_name == "search_jobs":
            return self.search_results.pop(0)
        if tool_name == "fetch_listing":
            return {
                "verified": True,
                "active": True,
                "final_url": "https://studentposao.hr/oglas/test/1",
            }
        return {"error": "neocekivani alat"}

    def log(self, event):
        self.events.append(event)


class _HistoryExecutor:
    def __init__(self):
        self.calls = []
        self.events = []
        self.listings = [
            {"title": f"Oglas {index}", "final_url": f"https://studentposao.hr/oglas/test/{index}"}
            for index in range(1, 4)
        ]

    def call(self, tool_name, **arguments):
        self.calls.append((tool_name, arguments))
        if tool_name == "task_state":
            return {"listings": self.listings}
        if tool_name == "fetch_listing":
            return {
                "verified": True,
                "active": True,
                "title": "Osvjezeni oglas",
                "final_url": arguments["url"],
                "detected_wage_eur_h": 10,
                "working_hours": "10 sati tjedno",
            }
        if tool_name == "estimate_earnings":
            return {"period_earnings_eur": arguments["hourly_wage"] * arguments["hours_per_week"]}
        raise AssertionError(f"Neocekivani alat: {tool_name}")

    def log(self, event):
        self.events.append(event)


class _StatusExecutor:
    def __init__(self):
        self.calls = []
        self.events = []
        self.listings = [{
            "title": "Nosenje maskota, animacija djece - DALAL OBRT",
            "final_url": "https://studentposao.hr/oglas/test/maskote",
        }]

    def call(self, tool_name, **arguments):
        self.calls.append((tool_name, arguments))
        if tool_name == "task_state":
            return {"listings": self.listings}
        if tool_name == "fetch_listing":
            return {
                "verified": True,
                "active": True,
                "final_url": arguments["url"],
                "active_evidence": "Oglas je dostupan na provjerenoj stranici.",
            }
        raise AssertionError(f"Neocekivani alat: {tool_name}")

    def log(self, event):
        self.events.append(event)


class _SearchExecutor:
    def __init__(self, search_results):
        self.search_results = search_results
        self.search_queries = []
        self.events = []

    def call(self, tool_name, **arguments):
        if tool_name == "search_jobs":
            self.search_queries.append(arguments["query"])
            return self.search_results.pop(0)
        if tool_name == "fetch_listing":
            return {
                "verified": True,
                "active": True,
                "final_url": arguments["url"],
                "weekend_only": True,
                "content_excerpt": "Rad samo vikendom.",
            }
        raise AssertionError(f"Neocekivani alat: {tool_name}")

    def log(self, event):
        self.events.append(event)


class _FakeFunctionCall:
    def __init__(self, name, arguments):
        self.name = name
        self.args = arguments
        self.id = name + "-id"


class _FakeResponse:
    def __init__(self, function_calls, text=""):
        self.function_calls = function_calls
        self.text = text
        self.candidates = [type("Candidate", (), {"content": object()})()]


def _fake_google_modules(responses):
    class FunctionDeclaration:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Tool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AutomaticFunctionCallingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Content:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Part:
        @staticmethod
        def from_text(**kwargs):
            return kwargs

        @staticmethod
        def from_function_response(**kwargs):
            return kwargs

    class Client:
        def __init__(self, **kwargs):
            self.models = type("Models", (), {"generate_content": self.generate_content})()

        def generate_content(self, **kwargs):
            return responses.pop(0)

    genai_module = ModuleType("google.genai")
    genai_module.Client = Client
    types_module = ModuleType("google.genai.types")
    types_module.FunctionDeclaration = FunctionDeclaration
    types_module.Tool = Tool
    types_module.AutomaticFunctionCallingConfig = AutomaticFunctionCallingConfig
    types_module.GenerateContentConfig = GenerateContentConfig
    types_module.Content = Content
    types_module.Part = Part
    genai_module.types = types_module
    google_module = ModuleType("google")
    google_module.genai = genai_module
    return google_module, genai_module


if __name__ == "__main__":
    unittest.main()
