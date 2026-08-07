"""
Food Waste Analytics for Restaurants — Streamlit dashboard.

Phase 3 (EDA) is fully wired up: filters, KPIs, and charts over the
synthetic/real waste_data.csv.

Phase 6-8 skeleton (GenAI layer + chat) is stubbed out in the sidebar tab
"Ask the analyst" — plug in your chosen LLM API in `call_llm()` once you've
picked a provider. The rest of the app (data loading, filtering, aggregation)
does not depend on that choice.

Run:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Food Waste Analytics", layout="wide")

DATA_PATH = "data/waste_data.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def build_structured_summary(df: pd.DataFrame) -> dict:
    """
    Phase 6 building block: turn filtered data into a compact, pre-computed
    JSON-able summary. This — NOT the raw dataframe — is what gets sent to
    the LLM later. Keeps numbers grounded and avoids hallucination.
    """
    if df.empty:
        return {}
    top_dish = df.groupby("dish_name")["wasted_qty"].sum().idxmax()
    top_dish_qty = df.groupby("dish_name")["wasted_qty"].sum().max()
    return {
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "total_waste_qty": round(df["wasted_qty"].sum(), 1),
        "total_waste_cost": round(df["waste_cost"].sum(), 2),
        "top_wasted_dish": top_dish,
        "top_wasted_dish_qty": round(top_dish_qty, 1),
        "avg_waste_weekday": round(df[~df["is_weekend"]]["wasted_qty"].mean(), 2),
        "avg_waste_weekend": round(df[df["is_weekend"]]["wasted_qty"].mean(), 2),
        "avg_waste_rainy": round(df[df["weather"] == "rainy"]["wasted_qty"].mean(), 2)
        if (df["weather"] == "rainy").any() else None,
    }


def call_llm(question: str, summary: dict) -> str:
    """
    Placeholder for Phase 6/7. Swap this out once you pick an LLM API
    (OpenAI / Anthropic / Gemini). The pattern stays the same regardless
    of provider:

        system_prompt = "You are a restaurant operations analyst. Only use
        the numbers provided below. Never invent figures."
        context = json.dumps(summary)
        response = <provider_client>.chat(system_prompt, context, question)
        return response.text

    For now this returns a canned response so the UI is testable end to end.
    """
    return (
        "[LLM not yet connected] Based on the current filters, total waste "
        f"cost is ₹{summary.get('total_waste_cost', 0):,.2f}, and "
        f"'{summary.get('top_wasted_dish', 'N/A')}' is the top wasted dish. "
        "Wire up your chosen LLM API in call_llm() to get a real narrative answer here."
    )


# ---------- Load & filter ----------
df = load_data("waste_data.csv")

st.sidebar.header("Filters")
date_min, date_max = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.date_input("Date range", (date_min, date_max), min_value=date_min, max_value=date_max)
dishes = st.sidebar.multiselect("Dishes", sorted(df["dish_name"].unique()), default=list(df["dish_name"].unique()))
weather_filter = st.sidebar.multiselect("Weather", sorted(df["weather"].unique()), default=list(df["weather"].unique()))

mask = (
    (df["date"].dt.date >= date_range[0])
    & (df["date"].dt.date <= date_range[1])
    & (df["dish_name"].isin(dishes))
    & (df["weather"].isin(weather_filter))
)
fdf = df[mask]

st.title("Food waste analytics")
st.caption("Understand, predict, and (soon) explain restaurant food waste.")

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total waste (units)", f"{fdf['wasted_qty'].sum():,.0f}")
c2.metric("Total waste cost", f"₹{fdf['waste_cost'].sum():,.0f}")
c3.metric("Avg waste / day", f"{fdf.groupby('date')['wasted_qty'].sum().mean():,.1f}")
c4.metric("Top wasted dish", fdf.groupby("dish_name")["wasted_qty"].sum().idxmax() if not fdf.empty else "—")

tab1, tab2 = st.tabs(["Dashboard", "Ask the analyst"])

with tab1:
    left, right = st.columns(2)

    with left:
        trend = fdf.groupby("date", as_index=False)["wasted_qty"].sum()
        fig = px.line(trend, x="date", y="wasted_qty", title="Waste over time")
        st.plotly_chart(fig, use_container_width=True)

        by_weekday = fdf.groupby("day_of_week", as_index=False)["wasted_qty"].mean()
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_weekday["day_of_week"] = pd.Categorical(by_weekday["day_of_week"], categories=order, ordered=True)
        by_weekday = by_weekday.sort_values("day_of_week")
        fig2 = px.bar(by_weekday, x="day_of_week", y="wasted_qty", title="Avg waste by day of week")
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        by_dish = fdf.groupby("dish_name", as_index=False)["wasted_qty"].sum().sort_values("wasted_qty", ascending=False)
        fig3 = px.bar(by_dish, x="wasted_qty", y="dish_name", orientation="h", title="Total waste by dish")
        st.plotly_chart(fig3, use_container_width=True)

        by_weather = fdf.groupby("weather", as_index=False)["wasted_qty"].mean()
        fig4 = px.bar(by_weather, x="weather", y="wasted_qty", title="Avg waste by weather")
        st.plotly_chart(fig4, use_container_width=True)

with tab2:
    st.write("Ask a question about the filtered data. This will be grounded in "
             "pre-computed numbers once an LLM API is connected (Phase 6-7).")
    summary = build_structured_summary(fdf)
    with st.expander("Structured summary sent to the LLM (debug view)"):
        st.json(summary)
    question = st.text_input("e.g. 'Why did we waste so much this week?'")
    if st.button("Ask") and question:
        answer = call_llm(question, summary)
        st.markdown(answer)
