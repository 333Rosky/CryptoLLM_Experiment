# Crypto Research

Explorations et notebooks pour stratégies crypto (ex: BTC). Point d'entrée recommandé: `btc_strategy.ipynb`.

## Features

- Modular package (`src/llm_crypto_fund/`) with Typer CLI (`lc run`, `lc show`)
- CCXT OHLCV ingestion with yfinance fallback and parquet caching
- Signal stack: 7d/30d returns, 30d annualised vol, RSI(14), BTC correlation
- LLM policy via direct HTTP call to `/v1/chat/completions` + Pydantic validation
- Risk overlays: weight caps, minimum holdings, stablecoin floor, turnover limits
- Portfolio ledger (Parquet), NAV tracking, Markdown daily reports
- Tooling: PEP 621 `pyproject.toml`, pre-commit, Ruff/Black/Isort, pytest, mypy
- GitHub Actions CI and scheduled daily workflow

## Project Layout

```
Crypto-Research/
├── btc_strategy.ipynb     # Notebook de stratégie BTC
├── src/crypto_research/   # (réservé à du code réutilisable si nécessaire)
└── README.md
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
jupyter lab  # ou jupyter notebook
```

## Notebooks

- `btc_strategy.ipynb` contient:
  - téléchargement de l’historique BTC (ccxt/yfinance)
  - métriques de base (retours, volatilité, RSI)
  - backtest d’une stratégie MA(50/200)

## Tooling

- `make dev` – install project with development extras
- `make fmt` / `make lint` / `make type` / `make test`
- Pre-commit hooks cover Ruff, Black, Isort, YAML hygiene
- GitHub Actions `ci.yml` runs linting, typing, and tests on PRs
- `daily.yml` executes `lc run` at 00:05 UTC and pushes updated logs/reports (requires `OPENAI_API_KEY` secret)

## Testing

```bash
pytest
mypy src
ruff check .
```

## Docker

Non prioritaire à ce stade (workflow focalisé sur notebooks).

## License

MIT © 2024
