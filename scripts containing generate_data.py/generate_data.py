"""
Synthetic data generator for Food Waste Analytics for Restaurants.

Simulates daily prep/sales/waste records for a set of dishes over N months,
with realistic built-in relationships:
  - Weekends have higher footfall (more prepped, more sold, but also more waste)
  - Rainy days increase waste (fewer walk-ins than expected -> over-prep)
  - Promotions/events increase both sales and waste (over-prep for anticipated demand)
  - Festivals cause waste spikes (large over-prep, uneven demand)
  - Some dishes are inherently more "wasteful" (perishable items, e.g. salads, seafood)

Run:
    python generate_data.py --months 9 --out data/waste_data.csv
"""

import argparse
import numpy as np
import pandas as pd
from datetime import timedelta

RNG = np.random.default_rng(42)

DISHES = [
    # (name, category, base_daily_demand, cost_per_unit, perishability)
    ("Paneer Butter Masala", "main", 40, 120, 0.35),
    ("Chicken Biryani", "main", 55, 150, 0.25),
    ("Veg Salad Bowl", "starter", 30, 60, 0.55),
    ("Fish Curry", "main", 25, 180, 0.50),
    ("Dal Tadka", "main", 45, 70, 0.20),
    ("Butter Naan", "bread", 80, 20, 0.10),
    ("Gulab Jamun", "dessert", 35, 40, 0.30),
    ("Tomato Soup", "starter", 28, 45, 0.45),
    ("Mutton Rogan Josh", "main", 20, 220, 0.30),
    ("Fresh Fruit Custard", "dessert", 22, 55, 0.60),
]

WEATHER_OPTIONS = ["clear", "cloudy", "rainy", "hot"]
WEATHER_PROBS = [0.45, 0.25, 0.20, 0.10]

FESTIVAL_DATES_OFFSET_DAYS = []  # filled dynamically relative to start date


def simulate(months: int, start_date: str) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    n_days = int(months * 30.44)
    dates = [start + timedelta(days=i) for i in range(n_days)]

    # pick a few random "festival/holiday" days across the range for spikes
    festival_days = set(
        RNG.choice(n_days, size=max(2, months), replace=False).tolist()
    )

    rows = []
    for day_idx, date in enumerate(dates):
        weekday = date.weekday()  # 0=Mon ... 6=Sun
        is_weekend = weekday >= 5
        is_festival = day_idx in festival_days
        weather = RNG.choice(WEATHER_OPTIONS, p=WEATHER_PROBS)
        is_rainy = weather == "rainy"
        has_promo = RNG.random() < (0.15 if not is_weekend else 0.25)

        for name, category, base_demand, cost, perishability in DISHES:
            # --- Expected demand (what customers actually order) ---
            demand_multiplier = 1.0
            if is_weekend:
                demand_multiplier *= 1.35
            if is_festival:
                demand_multiplier *= 1.6
            if has_promo:
                demand_multiplier *= 1.25
            if is_rainy:
                demand_multiplier *= 0.85  # fewer walk-ins on rainy days

            noise = RNG.normal(1.0, 0.12)
            sold_qty = max(0, base_demand * demand_multiplier * noise)

            # --- Prepared quantity (kitchen's estimate, imperfect) ---
            # Kitchens tend to over-prep, especially on weekends/festivals/promos
            # (anticipating demand that doesn't always materialize, especially if rainy)
            overprep_factor = 1.0 + 0.10 * perishability  # more perishable -> more caution buffer paradoxically
            if is_weekend:
                overprep_factor += 0.15
            if is_festival:
                overprep_factor += 0.30
            if has_promo:
                overprep_factor += 0.20
            if is_rainy:
                # kitchens often don't adjust down enough for rain -> extra waste
                overprep_factor += 0.15

            prep_noise = RNG.normal(1.0, 0.08)
            prepared_qty = sold_qty * overprep_factor * prep_noise
            prepared_qty = max(prepared_qty, sold_qty)  # can't sell more than prepared

            wasted_qty = max(0.0, prepared_qty - sold_qty)

            rows.append(
                {
                    "date": date.date().isoformat(),
                    "day_of_week": date.strftime("%A"),
                    "is_weekend": is_weekend,
                    "weather": weather,
                    "is_festival": is_festival,
                    "has_promotion": has_promo,
                    "dish_name": name,
                    "category": category,
                    "prepared_qty": round(prepared_qty, 1),
                    "sold_qty": round(sold_qty, 1),
                    "wasted_qty": round(wasted_qty, 1),
                    "cost_per_unit": cost,
                    "waste_cost": round(wasted_qty * cost, 2),
                }
            )

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=9, help="Number of months of data")
    parser.add_argument("--start-date", type=str, default="2025-01-01")
    parser.add_argument("--out", type=str, default="data/waste_data.csv")
    args = parser.parse_args()

    df = simulate(args.months, args.start_date)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Generated {len(df)} rows across {df['date'].nunique()} days "
          f"and {df['dish_name'].nunique()} dishes.")
    print(f"Saved to: {args.out}")
    print("\nQuick sanity check (waste by weather):")
    print(df.groupby("weather")["wasted_qty"].mean().round(2))
    print("\nQuick sanity check (waste by weekend vs weekday):")
    print(df.groupby("is_weekend")["wasted_qty"].mean().round(2))


if __name__ == "__main__":
    main()
