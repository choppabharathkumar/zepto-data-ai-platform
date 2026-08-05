# Zepto Data & AI Platform

## Overview

Zepto Data & AI Platform is an end-to-end Artificial Intelligence and Machine Learning capstone project. The repository contains three integrated modules that demonstrate the workflow of a real AI/ML engineer — from collecting raw data to building analytics and deploying an AI-powered support assistant.

| Module | Status | Marks |
| ------- | ------ | ----- |
| Data Pipeline | ✅ Complete | 25 |
| Analytics | 🚧 In Progress | 50 |
| Support Assistant | 🚧 In Progress | 25 |

---

## Repository Structure

```
zepto-data-ai-platform/
│
├── data_pipeline/
│   ├── data/
│   │   ├── raw/                  ← raw scraped CSV
│   │   └── processed/            ← cleaned, typed CSV
│   ├── sql/                      ← one .sql file per query
│   ├── outputs/
│   │   └── query_outputs/        ← one .txt file per query result
│   ├── books.db                  ← SQLite database
│   ├── constants.py
│   ├── utils.py
│   ├── scraper.py
│   ├── database_manager.py
│   ├── sql_queries.py
│   ├── main.py
│   ├── requirements.txt
│   └── README.md
│
├── analytics/
├── support_assistant/
├── .gitignore
└── README.md
```

---

## Modules

### 1. Data Pipeline

Scrapes book data from [books.toscrape.com](https://books.toscrape.com/),
cleans the data with pandas, stores it in a two-table normalised SQLite
database, and executes 6 SQL queries covering all required clauses.

**Fixed exchange rate:** `1 GBP = 105.50 INR` (project-defined constant,
no live API, no date reference — per the capstone specification).

#### Technologies

- Python · Requests · BeautifulSoup · Pandas · SQLite3

#### Run

```bash
# Install (from repo root)
pip install -r data_pipeline/requirements.txt

# Run
python data_pipeline/main.py
```

See [`data_pipeline/README.md`](data_pipeline/README.md) for the full
7-stage pipeline description, directory layout, SQL query index, and
design decisions.

---

### 2. Analytics

**Status: In Progress**

This module will contain:

- Titanic dataset cleaning & EDA
- Feature engineering
- Classification & regression models
- Model evaluation (accuracy, F1, ROC)
- Saved ML pipeline

---

### 3. Support Assistant

**Status: In Progress**

This module will contain:

- Offline-first RAG system with ChromaDB
- LangGraph-based retrieval & routing
- FastAPI REST interface
- Docker deployment

---

## Installation

Clone the repository:

```bash
git clone https://github.com/choppabharathkumar/zepto-data-ai-platform.git
cd zepto-data-ai-platform
```

Install dependencies for the module you want to run:

```bash
pip install -r data_pipeline/requirements.txt
```

---

## Running the Project

### Data Pipeline

```bash
python data_pipeline/main.py
```

Outputs are written to `data_pipeline/data/`, `data_pipeline/sql/`,
`data_pipeline/outputs/`, and `data_pipeline/books.db`.

Analytics and Support Assistant modules will be documented here when complete.

---

## Repository Workflow

Git is managed using a feature branch workflow:

- Feature branch created for each sprint.
- At least two commits per branch.
- Merged back into `main` when the sprint is complete.

---

## Author

**Choppa Bharath Kumar**

Artificial Intelligence and Machine Learning Capstone Project