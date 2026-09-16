"""Kalkulator procijenjene zarade."""

DEPENDENT_THRESHOLD_EUR = 3600.0
TAX_THRESHOLD_EUR = 12000.0


def estimate_earnings(hourly_wage, hours_per_week, weeks=52, earned_so_far=0):
    """Racuna zaradu za zadanu satnicu i broj sati rada."""
    hourly_wage = _positive_number(hourly_wage, "hourly_wage")
    hours_per_week = _positive_number(hours_per_week, "hours_per_week")
    weeks = _positive_number(weeks, "weeks")
    earned_so_far = _nonnegative_number(earned_so_far, "earned_so_far")

    period_earnings = round(hourly_wage * hours_per_week * weeks, 2)
    total_earnings = round(earned_so_far + period_earnings, 2)
    return {
        "hourly_wage_eur": hourly_wage,
        "hours_per_week": hours_per_week,
        "weeks": weeks,
        "earned_so_far_eur": earned_so_far,
        "period_earnings_eur": period_earnings,
        "total_earnings_eur": total_earnings,
        "thresholds_eur": {
            "dependent_status": DEPENDENT_THRESHOLD_EUR,
            "tax": TAX_THRESHOLD_EUR,
        },
        "dependent_threshold_reached": total_earnings >= DEPENDENT_THRESHOLD_EUR,
        "tax_threshold_reached": total_earnings >= TAX_THRESHOLD_EUR,
        "to_dependent_threshold_eur": round(
            max(0, DEPENDENT_THRESHOLD_EUR - total_earnings), 2
        ),
        "to_tax_threshold_eur": round(max(0, TAX_THRESHOLD_EUR - total_earnings), 2),
        "assumptions": (
            "Prikazani pragovi sluze za orijentacijski izracun i nisu porezno "
            "ili pravno tumacenje."
        ),
    }


def _positive_number(value, name):
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} mora biti veci od nule.")
    return number


def _nonnegative_number(value, name):
    number = float(value)
    if number < 0:
        raise ValueError(f"{name} ne moze biti negativan.")
    return number
