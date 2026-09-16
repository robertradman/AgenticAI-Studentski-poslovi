import unittest
from time import monotonic, sleep
from datetime import date, timedelta
from unittest.mock import patch

from fetch_listing import fetch_listing
import search_jobs


class SearchJobsSafetyTests(unittest.TestCase):
    def test_live_search_uses_a_bounded_verification_budget(self):
        self.assertEqual(search_jobs.MAX_LIVE_RESULTS, 6)
        self.assertEqual(search_jobs.MAX_STUDENTPOSAO_CANDIDATES, 4)
        self.assertEqual(search_jobs.MAX_EXTERNAL_CANDIDATES, 2)

    def test_rejects_category_and_article_urls(self):
        self.assertFalse(search_jobs._is_allowed_listing_url("https://www.posao.hr/gradovi/osijek/"))
        self.assertFalse(search_jobs._is_allowed_listing_url("https://www.posao.hr/clanci/vijesti/test/"))
        self.assertTrue(search_jobs._is_allowed_listing_url("https://www.posao.hr/oglasi/konobar/123/"))

    def test_rejects_inactive_listing_text(self):
        active, evidence = search_jobs._active_listing_status(
            "Ovaj oglas nije aktivan. Prijava na njega nije moguća."
        )
        self.assertFalse(active)
        self.assertIn("nije aktivan", evidence)

    def test_accepts_future_deadline(self):
        future = date.today() + timedelta(days=7)
        active, evidence = search_jobs._active_listing_status(
            f"Prijava do: {future.day:02d}. {future.month:02d}. {future.year}"
        )
        self.assertTrue(active)
        self.assertIn(future.isoformat(), evidence)

    def test_rejects_missing_student_signal(self):
        item = {
            "title": "Prodavač (m/ž)",
            "snippet": "Nedostaje: studentski",
            "content_excerpt": "Prodavač u Osijeku. Prijava do: 31. 12. 2099.",
            "active": True,
        }
        self.assertFalse(search_jobs._matches_live_filters(item, "studentski posao", "Osijek", None))

    def test_studentposao_domain_is_student_evidence(self):
        item = {
            "title": "Radnik u Osijeku",
            "snippet": "Prijava do: 31. 12. 2099.",
            "content_excerpt": "Posao u Osijeku.",
            "active": True,
            "source_domain": "studentposao.hr",
        }
        self.assertTrue(search_jobs._matches_live_filters(item, "studentski posao", "Osijek", None))

    def test_active_feed_confirms_only_unknown_activity(self):
        verification = {
            "active": False,
            "active_evidence": "Nema pozitivnog dokaza da je oglas aktivan.",
        }
        search_jobs._apply_active_feed_evidence({"listed_in_active_feed": True}, verification)
        self.assertTrue(verification["active"])

        expired = {
            "active": False,
            "active_evidence": "Rok prijave je prosao: 2020-01-01.",
        }
        search_jobs._apply_active_feed_evidence({"listed_in_active_feed": True}, expired)
        self.assertFalse(expired["active"])

    def test_min_wage_filter_requires_visible_wage(self):
        item = {
            "title": "Studentski posao",
            "snippet": "Rad u Osijeku preko studentskog ugovora.",
            "content_excerpt": "Prijava do: 31. 12. 2099.",
            "active": True,
        }
        self.assertFalse(search_jobs._matches_live_filters(item, "studentski posao", "Osijek", 8))

        item["content_excerpt"] += " Satnica 8.5 EUR/h."
        self.assertTrue(search_jobs._matches_live_filters(item, "studentski posao", "Osijek", 8))

    def test_wage_formats(self):
        self.assertEqual(search_jobs._extract_wage("Satnica 8,00 eura/h"), 8.0)
        self.assertEqual(search_jobs._extract_wage("Satnica 7,00 eur/sat"), 7.0)
        self.assertEqual(search_jobs._extract_wage("Satnica 6,56 €/h"), 6.56)
        self.assertIsNone(search_jobs._extract_wage("15 € po obavljenoj poslovnici"))

    def test_parses_current_studentposao_card_attribute_order(self):
        html = '''
        <div class="job-card relative" data-job-id="1">
          <a href="/oglas/test-osijek-1" class="absolute inset-0" aria-label="Test posao"></a>
          <p class="text-gray-500 text-sm">TVRTKA D.O.O.</p>
          <p class="text-gray-400 text-xs">📍 Osijek</p>
          <div class="text-sm font-bold text-primary">8,50 eura/h</div>
          <span class="text-xs text-gray-400">prije 3 h</span>
        </div>
        '''
        cards = search_jobs._parse_studentposao_cards(html, "https://studentposao.hr/?city=Osijek")

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["title"], "Test posao")
        self.assertEqual(cards[0]["url"], "https://studentposao.hr/oglas/test-osijek-1")
        self.assertIn("8,50 eura/h", cards[0]["snippet"])

    def test_parses_card_category_and_filters_non_it_before_fetch(self):
        html = '''
        <div class="job-card relative" data-job-id="1">
          <a href="/oglas/test-it-1" class="absolute" aria-label="Praksa"></a>
          <span class="bg-blue-50">💻 IT i digitalno</span>
          <p class="text-gray-400 text-xs">📍 Zagreb</p>
        </div>
        '''
        card = search_jobs._parse_studentposao_cards(html, "https://studentposao.hr/?q=praksa")[0]
        self.assertEqual(card["category"], "it i digitalno")
        self.assertTrue(search_jobs._studentposao_card_matches(card, "Zagreb", None, "IT praksa"))
        card["category"] = "ugostiteljstvo"
        self.assertFalse(search_jobs._studentposao_card_matches(card, "Zagreb", None, "IT praksa"))

    def test_it_internship_uses_two_targeted_portal_searches(self):
        self.assertEqual(
            search_jobs._studentposao_list_urls("studentska IT praksa", None),
            ["https://studentposao.hr/?q=praksa", "https://studentposao.hr/?q=IT"],
        )

    def test_studentposao_url_uses_one_scoped_request(self):
        self.assertEqual(
            search_jobs._studentposao_list_url("studentski vikend", "Osijek"),
            "https://studentposao.hr/?city=Osijek&q=vikend",
        )
        self.assertEqual(
            search_jobs._studentposao_list_url("studentski posao", None),
            "https://studentposao.hr/?",
        )

    def test_studentposao_url_uses_wage_and_remote_filters(self):
        self.assertEqual(
            search_jobs._studentposao_list_url("studentski remote", "Osijek", 9),
            "https://studentposao.hr/?city=Osijek&min_salary=9&is_remote=true",
        )

    def test_general_llm_paraphrase_does_not_become_portal_keyword(self):
        self.assertEqual(
            search_jobs._studentposao_list_url(
                "studentski posao nadi bilo osijeku okolici placen vise", "Osijek", 9
            ),
            "https://studentposao.hr/?city=Osijek&min_salary=9",
        )

    def test_external_search_stays_within_remaining_domain_sources(self):
        query = search_jobs._build_external_query("studentski vikend", "Osijek", 9)
        self.assertIn("site:mojposao.hr", query)
        self.assertIn("site:studentski-servis.hr", query)
        self.assertIn("site:posao.hr", query)
        self.assertNotIn("site:studentposao.hr", query)
        self.assertIn("Osijek", query)
        self.assertIn("9 EUR/h", query)

    def test_card_wage_without_hour_suffix_is_not_discarded(self):
        card = {
            "title": "Online posao",
            "snippet": "Lokacija: Osijek | Satnica: 10 €",
        }
        self.assertTrue(search_jobs._studentposao_card_matches(card, "Osijek", 9))
        self.assertEqual(search_jobs._extract_card_wage("Satnica: 20 eura Objavljeno danas"), 20.0)

    def test_studentposao_min_wage_accepts_card_format_without_hour_suffix(self):
        item = {
            "title": "Posao", "snippet": "Osijek | Satnica: 20 eura", "content_excerpt": "Opis",
            "active": True, "source_domain": "studentposao.hr",
        }
        self.assertTrue(search_jobs._matches_live_filters(item, "studentski posao", "Osijek", 9))
        self.assertEqual(item["detected_wage_eur_h"], 20.0)

    def test_official_remote_and_weekend_filters_keep_matching_cards(self):
        remote_item = {
            "title": "Posao", "snippet": "Osijek", "content_excerpt": "Opis bez remote rijeci.",
            "active": True, "source_domain": "studentposao.hr", "listed_in_remote_feed": True,
        }
        self.assertTrue(search_jobs._matches_live_filters(remote_item, "studentski remote", "Osijek", None))

        weekend_item = {
            "title": "Posao", "snippet": "Osijek", "content_excerpt": "Opis rasporeda.",
            "active": True, "source_domain": "studentposao.hr", "listed_in_active_feed": True,
        }
        self.assertTrue(search_jobs._matches_live_filters(weekend_item, "studentski vikend", "Osijek", None))

    def test_live_search_keeps_direct_result_when_external_search_is_empty(self):
        listing = {"title": "Test", "url": "https://studentposao.hr/oglas/test/1"}
        with patch("search_jobs._search_studentposao", return_value=[listing]), \
             patch("ddgs.DDGS") as ddgs:
            ddgs.return_value.__enter__.return_value.text.return_value = []
            self.assertEqual(search_jobs._search_live("studentski posao", "Osijek", None), [listing])

    def test_listing_response_is_reused_by_fetch_listing(self):
        response = _FakeResponse(
            "https://studentposao.hr/oglas/test/1",
            '<html><title>Test oglas</title><meta name="robots" content="unavailable_after: 2099-12-31"></html>',
        )
        search_jobs._clear_listing_cache()
        with patch("search_jobs._request_listing", return_value=response) as request:
            verification = search_jobs._verify_listing(response.url)
            detail = fetch_listing(response.url)

        self.assertTrue(verification["verified"])
        self.assertTrue(detail["verified"])
        self.assertIsNone(verification["application_deadline"])
        self.assertIsNone(detail["application_deadline"])
        self.assertEqual(request.call_count, 1)

    def test_candidate_verification_is_parallel_and_keeps_order(self):
        candidates = [{"url": f"https://studentposao.hr/oglas/test/{index}"} for index in range(4)]

        def slow_verification(url):
            sleep(0.1)
            return {"verified": True, "final_url": url}

        with patch("search_jobs._verify_listing", side_effect=slow_verification):
            started = monotonic()
            results = search_jobs._verify_candidates(candidates)
            elapsed = monotonic() - started

        self.assertLess(elapsed, 0.28)
        self.assertEqual([item["url"] for item, _ in results], [item["url"] for item in candidates])

    def test_network_timeout_is_five_seconds(self):
        self.assertEqual(search_jobs.HTTP_TIMEOUT_SECONDS, 5)


class _FakeResponse:
    def __init__(self, url, text):
        self.url = url
        self.text = text
        self.status_code = 200


if __name__ == "__main__":
    unittest.main()
