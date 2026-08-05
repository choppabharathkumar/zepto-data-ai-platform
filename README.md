# Zepto Data & AI Platform

## Overview

Zepto Data & AI Platform is an end-to-end Artificial Intelligence and Machine Learning capstone project. The repository contains three integrated modules that demonstrate the workflow of a real AI/ML engineer, from collecting raw data to building analytics and deploying an AI-powered support assistant.

| Module | Status | Marks |
| ------- | ------ | ----- |
| Data Pipeline | ✅ Complete | 25 |
| Analytics | 🚧 In Progress | 50 |
| Support Assistant | 🚧 In Progress | 25 |

---

# Repository Structure

```
zepto-data-ai-platform/
│
├── analytics/
├── data_pipeline/
├── support_assistant/
├── .gitignore
└── README.md
```

---

# Modules

## 1. Data Pipeline

The Data Pipeline module scrapes book data from the Books to Scrape website, cleans the data, stores it in a relational SQLite database, and performs SQL and Pandas analysis.

### Technologies

- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite3

### Features

- Web scraping
- Pagination handling
- Multiple category scraping
- Data cleaning
- Currency conversion (GBP → INR)
- SQLite database
- SQL queries
- Pandas integration

### Run

```bash
cd data_pipeline
pip install -r requirements.txt
python main.py
```

---

## 2. Analytics

Status

```
In Progress
```

This module will contain:

- Data preprocessing
- Exploratory Data Analysis
- Machine Learning
- Model Evaluation
- Visualizations

---

## 3. Support Assistant

Status

```
In Progress
```

This module will contain:

- Retrieval-Augmented Generation (RAG)
- Document Search
- Vector Database
- Large Language Model Integration
- Question Answering

---

# Installation

Clone the repository

```bash
git clone https://github.com/choppabharathkumar/zepto-data-ai-platform.git
```

Move into the project

```bash
cd zepto-data-ai-platform
```

Install dependencies for the required module.

Example:

```bash
cd data_pipeline
pip install -r requirements.txt
```

---

# Running the Project

### Data Pipeline

```bash
cd data_pipeline
python main.py
```

Analytics and Support Assistant modules will be added in future commits.

---

# Design Decisions

### Data Pipeline

- Scraped data from Books to Scrape.
- Scraped at least three categories.
- Handled pagination automatically.
- Stored data in a normalized SQLite database.
- Used two related tables with Primary Key and Foreign Key.
- Implemented SQL queries.
- Verified SQL JOIN results using Pandas.

---

# Repository Workflow

Git was managed using a feature branch workflow.

- Created a feature branch.
- Made multiple commits.
- Merged into the main branch.

---

# Author

**Choppa Bharath Kumar**

Artificial Intelligence and Machine Learning Capstone Project