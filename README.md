# FOOD-WASTE-ANALYTICS-FOR-RESTAURANT
# 🍽️ Food Waste Analytics for Restaurants

**A GenAI-powered data science system that predicts, explains, and reduces restaurant food waste — combining a real ML forecasting pipeline with a conversational natural-language layer.**

> Traditional analytics does the calculation. GenAI does the communication. This project puts an intelligent, conversational front-end on top of a real, grounded data science pipeline — not a chatbot pretending to know numbers it never computed.

---

## 📌 What This Project Does

1. **Understands the past** — analyzes historical prep, sales, and waste data to find patterns (which dishes waste the most, on which days, under what conditions).
2. **Predicts the future** — forecasts expected waste for upcoming days using weather, day-of-week, and event signals, and estimates prep quantities.
3. **Explains itself in plain language** — a GenAI layer turns model output and EDA findings into a conversational report or answers manager questions directly, e.g. *"Why did we waste so much paneer this week?"*

---

## 🏗️ System Architecture

```
DATA LAYER
  Raw Data → Cleaning & Preprocessing → Exploratory Data Analysis
                              │
                              ▼
MACHINE LEARNING LAYER
  Feature Engineering → Predictive Model → Structured Summary Layer (JSON)
                              │
                              ▼
GENERATIVE AI LAYER
  LLM (reasons only over the JSON summary, never raw data) → Grounded NL Output
                              │
                              ▼
INTERFACE LAYER
  Chat Interface (Streamlit)   /   Auto-Generated Weekly Report
                              │
                              ▼
                     Restaurant Manager
```

**Critical design rule:** the LLM never sees raw, unaggregated data and is never asked to do arithmetic. All numbers are pre-computed in Python and passed to the LLM as a structured JSON summary — this is what prevents hallucinated figures, the single biggest failure mode in GenAI analytics projects.

*(See `docs/Food_Waste_Analytics_Block_Diagram.pdf` for the full annotated architecture diagram.)*

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Modeling | scikit-learn, XGBoost |
| LLM API | Anthropic Claude API (default), OpenAI or Gemini also supported |
| LLM orchestration (optional) | LangChain / LlamaIndex |
| App / interface | Streamlit |
| Storage | CSV / SQLite (student scope), PostgreSQL (extended scope) |
| Deployment (optional) | Streamlit Community Cloud, Render, Hugging Face Spaces |

---

## 📁 Project Structure

```
food-waste-analytics/
├── data/
│   ├── raw/                     # original / simulated raw data
│   └── processed/                # cleaned + feature-engineered data
├── notebooks/
│   ├── 01_data_simulation.ipynb
│   ├── 02_cleaning_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── src/
│   ├── data_prep.py              # cleaning + feature engineering functions
│   ├── model.py                  # train / load / predict functions
│   ├── summary.py                # builds the structured JSON context (Phase 6)
│   ├── genai.py                  # system prompt + LLM call wrapper
│   └── pipeline.py               # full end-to-end pipeline (Phase 7)
├── app/
│   └── streamlit_app.py          # chat interface (Phase 8)
├── reports/
│   └── weekly_report_generator.py # scheduled auto-report job
├── docs/
│   ├── data_dictionary.md
│   └── Food_Waste_Analytics_Block_Diagram.pdf
├── tests/
│   └── test_genai_qa.py          # QA test suite for LLM answer grounding
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup

### 1. Clone and create a virtual environment
```bash
git clone https://github.com/<your-username>/food-waste-analytics.git
cd food-waste-analytics
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
plotly
streamlit
anthropic
python-dotenv
joblib
```

### 3. Set up your API key
Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
```

---

## 🚀 Usage

### Generate / load the dataset
```bash
python src/data_prep.py --simulate --output data/raw/food_waste_data.csv
```

### Train the predictive model
```bash
python src/model.py --train --input data/processed/features.csv --output models/waste_model.pkl
```

### Run the chat interface
```bash
streamlit run app/streamlit_app.py
```
Then open `http://localhost:8501` and ask questions like:
- *"Which dish wastes the most money?"*
- *"Is waste higher on rainy days?"*
- *"What should I prep less of this weekend?"*

### Generate a weekly report (manual run)
```bash
python reports/weekly_report_generator.py
```

### Schedule the weekly report (cron example)
```bash
# Runs every Monday at 8am
0 8 * * 1 /path/to/venv/bin/python /path/to/reports/weekly_report_generator.py
```

---

## 🧪 Testing the GenAI Layer

Every LLM answer is checked for grounding — i.e., every number in the response must trace back to the structured JSON summary, never invented.

```bash
python -m pytest tests/test_genai_qa.py -v
```

`tests/test_genai_qa.py` runs a fixed set of manager questions through the pipeline and checks:
- ✅ No numbers appear in the answer that aren't present in the JSON context
- ✅ The model correctly declines when asked something the data doesn't cover
- ✅ Cost/percentage translations are consistent with the underlying numbers

---

## 📊 Example Output

**Manager asks:** *"Why did we waste so much paneer this week?"*

**System responds:**
> Paneer Tikka accounted for ₹8,400 in waste cost this week — the highest of any dish. The main driver was Saturday's over-prep: 68 units were made but only 41 sold, largely because it was a rainy weekend (average waste on rainy days is 40% higher than sunny days across all dishes). Reducing Saturday prep by ~20 units would likely have saved roughly ₹1,500 this week alone.

*(Every number above is traceable to the pre-computed JSON summary — nothing is invented by the model.)*

---

## 📈 Model Performance

| Metric | Baseline (Linear Regression) | Final Model (XGBoost) |
|---|---|---|
| MAE | *fill in after training* | *fill in after training* |
| RMSE | *fill in after training* | *fill in after training* |

*(Update this table with your actual results — this is a required section for the portfolio writeup.)*

---

## 💰 Business Impact

- Estimated **X%** reduction in weekly food waste cost based on model-informed prep recommendations
- Estimated **₹X / month** in potential savings (see `reports/business_impact.md` for the calculation)

*(Fill in with your actual numbers from Phase 9 of the build.)*

---

## 🗺️ Roadmap / Stretch Goals

- [ ] RAG over unstructured chef notes / supplier remarks for qualitative reasoning
- [ ] Prescriptive engine: exact prep-quantity recommendations with confidence ranges
- [ ] Multi-restaurant support with cross-location benchmarking
- [ ] CO₂-equivalent sustainability metric (ESG angle)
- [ ] Few-shot prompt tuning for more consistent tone/recommendation style

---

## ⚠️ Design Principle (Read Before Contributing)

> **Never pass raw dataframes or unaggregated rows into an LLM prompt.**

All numeric computation happens in Python (`src/summary.py`). The LLM (`src/genai.py`) only ever receives a small, pre-verified JSON object and is explicitly instructed never to invent numbers. Any contribution that routes raw data directly into a prompt breaks the core design guarantee of this project and will not be merged.

---

## 📄 License

MIT License — feel free to fork and adapt for your own portfolio project.

---

## 🙋 Author

Built as a data science + GenAI portfolio project demonstrating an end-to-end pipeline: EDA → predictive modeling → LLM-grounded natural language interface.
