# Food Waste Analytics for Restaurants

A Streamlit dashboard for understanding, exploring, and (soon) explaining restaurant food waste — built on daily prep/sales/waste records across a menu of dishes.

## What's in this repo

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard: filters, KPIs, charts, and a stubbed-out "Ask the analyst" chat tab |
| `generate_data.py` | Synthetic data generator that produces `waste_data.csv` |
| `waste_data.csv` | Daily records per dish: prepared/sold/wasted quantities, cost, weather, weekend/festival/promo flags |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

## Generating the data (optional — a copy is already included)

```bash
python generate_data.py --months 9 --start-date 2025-01-01 --out data/waste_data.csv
```

The generator simulates 10 dishes over N months with realistic patterns baked in:
- **Weekends** → higher footfall, more prepped, more sold, but also more waste
- **Rainy days** → fewer walk-ins than kitchens expect → over-prep → more waste
- **Promotions/events** → boost both sales and waste (over-prep for anticipated demand)
- **Festivals** → large waste spikes from uneven, hard-to-predict demand
- **Perishability** → some dishes (salads, seafood, custards) waste more inherently

Each row = one dish on one day, with `prepared_qty`, `sold_qty`, `wasted_qty`, `cost_per_unit`, and `waste_cost`.

## Running the dashboard

The app expects the data at `data/waste_data.csv` relative to where you run it:

```bash
mkdir -p data
cp waste_data.csv data/waste_data.csv   # if not already generated there
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## What the dashboard does today (Phase 3 — EDA, fully wired)

- **Filters**: date range, dish, weather
- **KPIs**: total waste (units), total waste cost, avg waste/day, top wasted dish
- **Charts**:
  - Waste over time (line)
  - Avg waste by day of week (bar)
  - Total waste by dish (horizontal bar)
  - Avg waste by weather (bar)

## What's stubbed out (Phase 6–8 — GenAI layer)

The **"Ask the analyst"** tab lets a user type a free-text question about the filtered data. Right now:

1. `build_structured_summary()` turns the filtered dataframe into a compact, pre-computed JSON summary (date range, total waste qty/cost, top wasted dish, weekday vs. weekend avg, rainy-day avg). This — **not the raw dataframe** — is designed to be what gets sent to an LLM, to keep answers grounded and avoid hallucinated numbers.
2. `call_llm()` is a placeholder that returns a canned response using the summary. Swap in a real LLM call once a provider is chosen (OpenAI / Anthropic / Gemini) — the pattern is the same regardless of provider:
   ```python
   system_prompt = "You are a restaurant operations analyst. Only use the numbers provided below. Never invent figures."
   context = json.dumps(summary)
   response = <provider_client>.chat(system_prompt, context, question)
   return response.text
   ```
3. A debug expander in the UI shows the exact structured summary that would be sent to the LLM, so you can sanity-check grounding before wiring up a real API.

## Next steps

- Pick an LLM provider and implement `call_llm()`
- Decide whether the chat should support follow-up/multi-turn questions (would need conversation history passed alongside the summary)
- Optionally add a forecasting tab (the `scikit-learn` / `xgboost` deps are already in `requirements.txt` for this)
