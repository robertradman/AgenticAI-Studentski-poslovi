import argparse
import sys

from search_jobs import search_jobs


def main(argv):
    parser = argparse.ArgumentParser(description="Provjera live search_jobs alata.")
    parser.add_argument("query")
    parser.add_argument("--city")
    parser.add_argument("--min-wage", type=float)
    parser.add_argument(
        "--require-results",
        action="store_true",
        help="Vrati gresku ako nema nijednog provjerenog oglasa.",
    )
    args = parser.parse_args(argv)

    results = search_jobs(args.query, city=args.city, min_wage=args.min_wage)
    for listing in results:
        assert listing.get("verified") is True, listing
        assert listing.get("active") is True, listing
        assert listing.get("final_url"), listing
        assert listing.get("source_domain"), listing

    print(f"Broj provjerenih oglasa: {len(results)}")
    if not results:
        print("Nema rezultata koji zadovoljavaju strogu provjeru.")
        return 1 if args.require_results else 0

    for index, listing in enumerate(results, start=1):
        print(f"{index}. {listing.get('title')}")
        print(f"   Izvor: {listing.get('source_domain')}")
        print(f"   Link: {listing.get('final_url')}")
        print(f"   Dokaz aktivnosti: {listing.get('active_evidence')}")
        print(f"   Rok prijave: {listing.get('application_deadline') or 'nije pronaden'}")

    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main(sys.argv[1:]))
