# hkk-regi-lap-classification

HKK old card classification project

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) `https://astral.sh/uv/` for dependency management
- Local MySQL database with card base data (not included in this project)

## Setup

Install dependencies:

```bash
uv sync
```

## Scripts

| Script | Description |
|---|---|
| `01_get_card_flags.py` | Fetch card base data and flags from the local card-search database |
| `02_multi_class_classifier.py` | Multi-label classifier: tests 5 ML algorithms to predict flags for older cards |
| `03_crowdsource_classifier.py` | Flask webapp for crowdsourced card classification (ML prediction + human votes) |

### Run scripts

```bash
uv run python 01_get_card_flags.py
uv run python 02_multi_class_classifier.py
uv run python 03_crowdsource_classifier.py
```

## Configuration

For database connection, create a `.env` file in the project root with the following content or run `cp .env.example .env` and edit the values as your environment requires:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=secret
DB_NAME=hkk-lapkereso-db
```

> Note: only required for `01_get_card_flags.py`

## Data files

| File | Description |
|---|---|
| `card_flags.csv` | Card base data with flags (output of script 01) |
| `card_flags_predicted.csv` | ML-predicted flags for older cards (output of script 02) |
| `crowdsource.db` | SQLite database for the crowdsource webapp (created by script 03) |

## Dependencies

| Package | Purpose |
|---|---|
| `mysql-connector-python` | MySQL database connectivity |
| `requests` | HTTP requests |
| `pandas` | Data manipulation |
| `scikit-learn` | ML algorithms |
| `xgboost` | XGBoost classifier |
| `lightgbm` | LightGBM classifier |
| `flask` | Crowdsource webapp |

## Crowdsource webapp (`03_crowdsource_classifier.py`)

A card is "closed" when `VOTES_REQUIRED` matching votes are received (AI prediction counts as one vote).
This threshold is configurable at the top of the script:

```python
VOTES_REQUIRED = 2  # set to 2 or 3
```
