# Zepto Data & AI Platform

Capstone project: an end-to-end data + AI platform with three linked modules.

| Module | Status | Marks |
|---|---|---|
| [`/data_pipeline`](./data_pipeline/README.md) | Complete | 25 |
| `/analytics` | Not started yet | 50 |
| `/support_assistant` | Not started yet | 25 |

## Setup

Each module has its own `requirements.txt` (see that module's folder). Install
per-module as you work on it, e.g.:

```bash
cd data_pipeline
pip install -r requirements.txt
```

## Running each module

- **data_pipeline**: `cd data_pipeline && python main.py` — see
  [`data_pipeline/README.md`](./data_pipeline/README.md) for full details,
  design decisions, and the SQL query list.
- **analytics**: not yet built.
- **support_assistant**: not yet built.

## Design decisions

See each module's own README for its specific design decisions. The
`data_pipeline` module's are documented in
[`data_pipeline/README.md`](./data_pipeline/README.md#design-decisions).
