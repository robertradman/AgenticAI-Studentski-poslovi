"""Pokretanje skupa primjera i spremanje rezultata provjere."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from agent import run_task

ROOT_DIR = os.path.dirname(__file__)
DEFAULT_REPORT = os.path.join(ROOT_DIR, "reports", "evaluation_report.json")
EVALUATION_TASKS = [
    "Pronadi studentske poslove u Osijeku koji placaju vise od 8 EUR/h i ne traze iskustvo.",
    "Jesu li otvorene ljetne IT prakse u Zagrebu s rokom prijave nakon 1. rujna?",
    "Usporedi 3 najnovija ugostiteljska posla u Osijeku po satnici i satima tjedno.",
    "Pronadi studentske poslove koji se rade iskljucivo vikendom.",
    "Ako radim cijelu godinu 20 sati tjedno za 7 EUR/h, kolika je zarada i jesam li blizu pragova?",
    "Pronadi studentske oglase objavljene zadnja 3 dana za remote rad.",
    "Daj kratak pregled 3 najbolje placena oglasa za sankera u Osijeku ovaj tjedan.",
    'Provjeri je li oglas "Zamjena nastavnika matematike u srednjoj skoli" jos aktivan.',
    "Zapamti da gledam samo IT poslove u Osijeku.",
    "Od zadnja 3 oglasa koja sam vidio, koji ima najraniji rok prijave?",
]


def evaluate(mode="llm", report_path=DEFAULT_REPORT):
    results = []
    for number, query in enumerate(EVALUATION_TASKS, 1):
        run = run_task(query, mode=mode)
        trace = run["trace"]
        calls = trace["tool_calls"]
        results.append({
            "id": number,
            "query": query,
            "outcome": run["outcome"],
            "route": (run["plan"] or {}).get("route"),
            "total_steps": trace["total_steps"],
            "tools": [call["tool"] for call in calls],
            "answer": run["answer"],
            "passed_contract": _passes_contract(run, calls),
        })
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "task_count": len(results),
        "contract_passed": sum(item["passed_contract"] for item in results),
        "results": results,
        "note": "no_verified_results i needs_history su valjani transparentni ishodi za promjenjivi web ili nedostatak prethodne povijesti; nisu dokaz da je oglas pronaden.",
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2, default=str)
    return report


def _passes_contract(run, calls):
    outcome = run["outcome"]
    if outcome in {"error", "needs_llm"}:
        return False
    if run["plan"] is None:
        return False
    if outcome == "completed" and not calls:
        return False
    return all(call["tool"] in {
        "search_jobs", "fetch_listing", "estimate_earnings", "task_state", "generate_application",
    } for call in calls)


def main(argv):
    parser = argparse.ArgumentParser(description="Provjera skupa korisnickih upita.")
    parser.add_argument("--mode", choices=["heuristic", "llm", "auto"], default="llm")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    report = evaluate(args.mode, args.report)
    print(f"Izvjestaj: {args.report}")
    print(f"Ugovorni prolaz: {report['contract_passed']}/{report['task_count']}")
    for result in report["results"]:
        print(f"{result['id']}. {result['outcome']} | {result['total_steps']} koraka | {', '.join(result['tools'])}")
    return 0 if report["contract_passed"] == report["task_count"] else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main(sys.argv[1:]))
