<div align="center">

# 🍽️ Food Waste Analytics for Restaurants

**A GenAI-powered data science system that predicts, explains, and reduces restaurant food waste.**

*A real ML forecasting pipeline — with a grounded conversational layer on top, not a chatbot guessing numbers.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-2E8B57)
![Streamlit](https://img.shields.io/badge/Streamlit-Interface-FF4B4B?logo=streamlit&logoColor=white)
![Claude API](https://img.shields.io/badge/Claude%20API-GenAI%20Layer-6C3483)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

> **The core idea:** traditional ML/analytics does the calculation. GenAI does the communication. This project puts an intelligent, conversational front-end on top of a real analytics engine — every number the AI mentions is traceable back to code that computed it, never invented.

---

## 📋 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [Why This Project](#-why-this-project)
- [System Architecture](#️-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup](#️-setup)
- [Usage](#-usage)
- [Example Interaction](#-example-interaction)
- [Testing the GenAI Layer](#-testing-the-genai-layer)
- [Model Performance](#-model-performance)
- [Business Impact](#-business-impact)
- [Roadmap](#️-roadmap--stretch-goals)
- [Core Design Principle](#️-core-design-principle-read-before-contributing)
- [License](#-license)

---

## 🎯 What This Project Does

| Capability | Description |
|---|---|
| **Understands the past** | Analyzes historical prep, sales, and waste data to find patterns — which dishes waste the most, on which days, under what conditions |
| **Predicts the future** | Forecasts expected waste for upcoming days using weather, day-of-week, and event signals |
| **Explains itself** | A GenAI layer turns raw numbers into a conversational report a manager can read or query directly — e.g. *"Why did we waste so much paneer this week?"* |

---

## 💡 Why This Project

Food waste is one of the most measurable inefficiencies in the restaurant industry — restaurants routinely over-prepare to avoid running out, which leads directly to spoilage and financial loss. This isn't a toy problem; it maps to real cost savings and operational decisions restaurant owners care about daily.

| Typical Student Project | This Project |
|---|---|
| Static dashboard with charts | Conversational, GenAI-explained insights |
| One model, one metric shown | Full pipeline: EDA → model → LLM narrative → recommendation |
| Generic public dataset (Titanic, Iris) | Domain-specific, business-relevant data |
| "Here's the accuracy score" | "Here's how much money this saves the business" |
| No natural language interface | Manager can literally ask questions and get grounded answers |

The differentiator isn't the ML model — regression on tabular data is well-trodden ground. The differentiator is **applied GenAI engineering**: grounding an LLM in real computed data instead of letting it hallucinate numbers.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                  │
│  Raw Data → Cleaning & Preprocessing → EDA                   │
└───────────────────────────┬───────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  MACHINE LEARNING LAYER                                       │
│  Feature Engineering → Predictive Model → JSON Summary Layer  │
└───────────────────────────┬───────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  GENERATIVE AI LAYER                                          │
│  LLM reasons ONLY over the JSON summary → Grounded NL Output  │
│  (never sees raw data, never does arithmetic itself)          │
└───────────────────────────┬───────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  INTERFACE LAYER                                               │
│  Chat Interface (Streamlit)   ⇆   Auto-Generated Weekly Report│
└───────────────────────────┬───────────────────────────────────┘
                             ▼
                    👤 Restaurant Manager
```

> 🔒 **Critical design rule:** the LLM never sees raw, unaggregated data and is never asked to do arithmetic. All numbers are pre-computed in Python and passed to the LLM as a structured JSON summary — this is what prevents hallucinated figures, the single biggest failure mode in GenAI analytics projects.

📎 Full annotated diagram: [`docs/Food_Waste_Analytics_Block_Diagram.pdf`](docs/Food_Waste_Analytics_Block_Diagram.pdf)

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Modeling | `scikit-learn`, `XGBoost` |
| LLM API | Anthropic Claude API *(default)* — OpenAI or Gemini also supported |
| LLM orchestration *(optional)* | LangChain / LlamaIndex |
| App / interface | Streamlit |
| Storage | CSV / SQLite (student scope) → PostgreSQL (extended scope) |
| Deployment *(optional)* | Streamlit Community Cloud, Render, Hugging Face Spaces |

---

## 📁 Project Structure

```
food-waste-analytics/
├── data/
│   ├── raw/                       # original / simulated raw data
│   └── processed/                 # cleaned + feature-engineered data
├── notebooks/
│   ├── 01_data_simulation.ipynb
│   ├── 02_cleaning_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── src/
│   ├── data_prep.py               # cleaning + feature engineering
│   ├── model.py                   # train / load / predict
│   ├── summary.py                 # builds structured JSON context
│   ├── genai.py                   # system prompt + LLM call wrapper
│   └── pipeline.py                # full end-to-end pipeline
├── app/
│   └── streamlit_app.py           # chat interface
├── reports/
│   ├── weekly_report_generator.py # scheduled auto-report job
│   └── business_impact.md         # cost-savings calculation
├── docs/
│   ├── data_dictionary.md
│   └── Food_Waste_Analytics_Block_Diagram.pdf
├── tests/
│   └── test_genai_qa.py           # grounding QA suite for LLM answers
├── models/
│   └── waste_model.pkl            # trained model artifact (gitignored)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

**1. Clone and create a virtual environment**
```bash
git clone https://github.com/<your-username>/food-waste-analytics.git
cd food-waste-analytics
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

<details>
<summary><b>requirements.txt</b> (click to expand)</summary>

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
pytest
```
</details>

**3. Configure your API key**

Copy the example env file and add your key:
```bash
cp .env.example .env
```
```
# .env
ANTHROPIC_API_KEY=your_key_here
```

---

## 🚀 Usage

| Step | Command |
|---|---|
| Generate / simulate dataset | `python src/data_prep.py --simulate --output data/raw/food_waste_data.csv` |
| Train the predictive model | `python src/model.py --train --input data/processed/features.csv --output models/waste_model.pkl` |
| Launch the chat interface | `streamlit run app/streamlit_app.py` |
| Generate a weekly report manually | `python reports/weekly_report_generator.py` |

Once the Streamlit app is running, open **`http://localhost:8501`** and try:
- *"Which dish wastes the most money?"*
- *"Is waste higher on rainy days?"*
- *"What should I prep less of this weekend?"*

**Schedule the weekly report** (cron example — runs every Monday at 8am):
```bash
0 8 * * 1 /path/to/venv/bin/python /path/to/reports/weekly_report_generator.py
```

---

## 💬 Example Interaction

**Manager asks:**
> "Why did we waste so much paneer this week?"

**System responds:**
> Paneer Tikka accounted for ₹8,400 in waste cost this week — the highest of any dish. The main driver was Saturday's over-prep: 68 units were made but only 41 sold, largely because it was a rainy weekend (average waste on rainy days is 40% higher than sunny days across all dishes). Reducing Saturday prep by ~20 units would likely have saved roughly ₹1,500 this week alone.

✅ *Every number above is traceable to the pre-computed JSON summary — nothing is invented by the model.*

---

## 🧪 Testing the GenAI Layer

Every LLM answer is checked for **grounding** — every number in the response must trace back to the structured JSON summary, never invented.

```bash
python -m pytest tests/test_genai_qa.py -v
```

`tests/test_genai_qa.py` runs a fixed set of manager questions through the pipeline and checks:
- ✅ No numbers appear in the answer that aren't present in the JSON context
- ✅ The model correctly declines when asked something the data doesn't cover
- ✅ Cost/percentage translations are consistent with the underlying numbers

---

## 📈 Model Performance

| Metric | Baseline (Linear Regression) | Final Model (XGBoost) |
|---|---|---|
| MAE | *fill in after training* | *fill in after training* |
| RMSE | *fill in after training* | *fill in after training* |

*(Update this table with your actual results — this is a required section for the portfolio writeup.)*

---

## 💰 Business Impact

- Estimated **X%** reduction in weekly food waste cost from model-informed prep recommendations
- Estimated **₹X / month** in potential savings — see [`reports/business_impact.md`](reports/business_impact.md) for the full calculation

*(Fill in with your actual numbers after running the evaluation phase.)*

---

## 🗺️ Roadmap / Stretch Goals

- [ ] RAG over unstructured chef notes / supplier remarks for qualitative reasoning
- [ ] Prescriptive engine — exact prep-quantity recommendations with confidence ranges
- [ ] Multi-restaurant support with cross-location benchmarking
- [ ] CO₂-equivalent sustainability metric (ESG angle)
- [ ] Few-shot prompt tuning for more consistent tone/recommendation style

---

## ⚠️ Core Design Principle (Read Before Contributing)

> **Never pass raw dataframes or unaggregated rows into an LLM prompt.**

All numeric computation happens in Python (`src/summary.py`). The LLM (`src/genai.py`) only ever receives a small, pre-verified JSON object and is explicitly instructed never to invent numbers. Any contribution that routes raw data directly into a prompt breaks the core design guarantee of this project and will not be merged.

---

## 📄 License

MIT License — feel free to fork and adapt for your own portfolio project.

---

<div align="center">

**Built as an end-to-end data science + GenAI portfolio project**
*EDA → predictive modeling → LLM-grounded natural language interface*

</div>
