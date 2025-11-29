"""Download historical US option data from Polygon.io."""

from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path
from math import erf, exp, log, sqrt

import pandas as pd
import requests


BASE_URL = "https://api.polygon.io"
API_KEY = "api_key"
REQUEST_SPACING_SECONDS = 13.0  # 5 calls/min allowance buffer
LAST_REQUEST_TS = 0.0
DEFAULT_UNDERLYING = "AAPL"
DEFAULT_MAX_CONTRACTS: int | None = None  # None = no limit
DEFAULT_RATE = 0.04
DEFAULT_DIV_YIELD = 0.0
DEFAULT_CONTRACT_TYPE = "all"
DEFAULT_YEARS = 2
LOG_EVERY_N_CONTRACTS = 20


def log(msg: str) -> None:
    print(f"[info] {msg}")


def default_date_range() -> tuple[str, str]:
    """Return (start_date, end_date) ISO strings for the last 30 days (ending yesterday)."""
    end = dt.date.today() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=30)
    return start.isoformat(), end.isoformat()


def get_with_retries(
    url: str,
    params: dict | None,
    attempts: int = 6,
    base_delay: float = REQUEST_SPACING_SECONDS,
) -> requests.Response:
    """GET with basic 429 handling."""
    last_response: requests.Response | None = None
    for attempt in range(attempts):
        sleep_for = REQUEST_SPACING_SECONDS - (time.time() - LAST_REQUEST_TS)
        if sleep_for > 0:
            time.sleep(sleep_for)
        response = requests.get(url, params=params if url.startswith(BASE_URL) else None, timeout=30)
        globals()["LAST_REQUEST_TS"] = time.time()
        if response.status_code == 403:
            # Plan limit (e.g., beyond 2y). Fail fast with message.
            response.raise_for_status()
        if response.status_code != 429:
            response.raise_for_status()
            return response
        last_response = response
        retry_after = response.headers.get("Retry-After")
        delay = max(base_delay, REQUEST_SPACING_SECONDS)
        if retry_after and retry_after.isdigit():
            delay = float(retry_after)
        time.sleep(delay)
    if last_response is not None:
        last_response.raise_for_status()
    raise RuntimeError("Failed after retries")


def require_api_key(cli_arg: str | None) -> str:
    api_key = cli_arg or os.getenv("POLYGON_API_KEY") or API_KEY
    if not api_key:
        raise RuntimeError("No API key. Provide --api-key or set POLYGON_API_KEY.")
    return api_key


def fetch_option_contracts(
    underlying: str,
    start_date: dt.date,
    end_date: dt.date,
    api_key: str,
    max_contracts: int | None,
    contract_type: str,
) -> list[dict]:
    """Pull a limited set of option tickers using multiple as_of snapshots."""
    total_days = (end_date - start_date).days or 1
    step = dt.timedelta(days=max(30, total_days // 6))
    max_snapshots = 8
    snapshots: list[dt.date] = []
    current = start_date
    while current <= end_date and len(snapshots) < max_snapshots:
        snapshots.append(current)
        current += step
    if snapshots and snapshots[-1] != end_date:
        snapshots.append(end_date)

    log(f"Fetching contracts for {underlying} over {len(snapshots)} snapshots...")
    seen_tickers: set[str] = set()
    contracts: list[dict] = []
    for as_of in snapshots:
        url = f"{BASE_URL}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "as_of": as_of.isoformat(),
            "limit": 1000,
            "sort": "expiration_date",
            "order": "asc",
            "apiKey": api_key,
        }
        if contract_type in {"call", "put"}:
            params["contract_type"] = contract_type

        while url and (max_contracts is None or len(contracts) < max_contracts):
            response = get_with_retries(url, params=params)
            payload = response.json()
            for option in payload.get("results", []):
                ticker = option.get("ticker")
                expiration = option.get("expiration_date")
                strike = option.get("strike_price")
                ctype = option.get("contract_type")
                if (
                    ticker
                    and expiration
                    and dt.date.fromisoformat(expiration) >= start_date
                    and ticker not in seen_tickers
                ):
                    seen_tickers.add(ticker)
                    contracts.append(
                        {
                            "ticker": ticker,
                            "occ_symbol": ticker[2:] if ticker.startswith("O:") else ticker,
                            "strike": strike,
                            "type": ctype,
                            "expiration": expiration,
                            "underlying": option.get("underlying_ticker") or underlying,
                        }
                    )
                    if max_contracts is not None and len(contracts) >= max_contracts:
                        break
            next_url = payload.get("next_url")
            if next_url and "apiKey=" not in next_url:
                next_url = f"{next_url}&apiKey={api_key}"
            url = next_url
            params = None
            if not url:
                break
        if max_contracts is not None and len(contracts) >= max_contracts:
            break

    log(f"Collected {len(contracts)} contracts for {underlying}.")
    return contracts


def fetch_underlying_closes(
    underlying: str, start_date: str, end_date: str, api_key: str
) -> dict[str, float]:
    url = f"{BASE_URL}/v2/aggs/ticker/{underlying}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
    response = get_with_retries(url, params=params)
    payload = response.json()
    prices: dict[str, float] = {}
    for bar in payload.get("results", []):
        date_iso = dt.datetime.fromtimestamp(bar["t"] / 1000, tz=dt.timezone.utc).date().isoformat()
        prices[date_iso] = bar.get("c")
    return prices


def fetch_option_bars(
    option: dict,
    start_date: str,
    end_date: str,
    api_key: str,
    underlying_closes: dict[str, float],
) -> list[dict]:
    option_ticker = option["ticker"]
    url = f"{BASE_URL}/v2/aggs/ticker/{option_ticker}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
    response = get_with_retries(url, params=params)
    payload = response.json()
    if payload.get("status") != "OK" or not payload.get("results"):
        return []

    rows: list[dict] = []
    for bar in payload["results"]:
        timestamp_ms = bar["t"]
        date_iso = dt.datetime.fromtimestamp(timestamp_ms / 1000, tz=dt.timezone.utc).date().isoformat()
        underlying_price = underlying_closes.get(date_iso)
        rows.append(
            {
                "ticker": option_ticker,
                "occ_symbol": option.get("occ_symbol"),
                "underlying": option.get("underlying"),
                "type": option.get("type"),
                "strike": option.get("strike"),
                "expiration": option.get("expiration"),
                "timestamp": timestamp_ms,
                "date": date_iso,
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v"),
                "vwap": bar.get("vw"),
                "transactions": bar.get("n"),
                "bid": None,
                "ask": None,
                "mid": None,
                "bid_size": None,
                "ask_size": None,
                "last_price": None,
                "last_size": None,
                "underlying_price": underlying_price,
                "implied_volatility": None,
                "open_interest": None,
            }
        )
    return rows


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return (1.0 / sqrt(2.0 * 3.141592653589793)) * exp(-0.5 * x * x)


def bs_price_iv(
    S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool
) -> float:
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return float("nan")
    sqrtT = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if is_call:
        return S * exp(-q * T) * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    return K * exp(-r * T) * norm_cdf(-d2) - S * exp(-q * T) * norm_cdf(-d1)


def bs_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> dict:
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return {k: float("nan") for k in ["delta", "gamma", "theta", "vega", "rho"]}
    sqrtT = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf = norm_pdf(d1)
    if is_call:
        delta = exp(-q * T) * norm_cdf(d1)
        theta = (
            -(S * exp(-q * T) * pdf * sigma) / (2 * sqrtT)
            - r * K * exp(-r * T) * norm_cdf(d2)
            + q * S * exp(-q * T) * norm_cdf(d1)
        )
        rho = K * T * exp(-r * T) * norm_cdf(d2)
    else:
        delta = -exp(-q * T) * norm_cdf(-d1)
        theta = (
            -(S * exp(-q * T) * pdf * sigma) / (2 * sqrtT)
            + r * K * exp(-r * T) * norm_cdf(-d2)
            - q * S * exp(-q * T) * norm_cdf(-d1)
        )
        rho = -K * T * exp(-r * T) * norm_cdf(-d2)
    gamma = exp(-q * T) * pdf / (S * sigma * sqrtT)
    vega = S * exp(-q * T) * pdf * sqrtT
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta / 365.0,  # per-day theta for readability
        "vega": vega / 100.0,  # per 1 vol point
        "rho": rho / 100.0,  # per 1 rate point
    }


def implied_vol(
    price: float, S: float, K: float, T: float, r: float, q: float, is_call: bool
) -> float | None:
    # Basic bounds check
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
    if price < intrinsic:
        return None
    low, high = 1e-4, 5.0
    for _ in range(30):
        mid = 0.5 * (low + high)
        mid_price = bs_price_iv(S, K, T, r, q, mid, is_call)
        if not mid_price or mid_price != mid_price:  # NaN guard
            return None
        if abs(mid_price - price) < 1e-4:
            return mid
        if mid_price > price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def add_greeks(df: pd.DataFrame, rate: float, div_yield: float) -> pd.DataFrame:
    greeks_data = {
        "implied_volatility": [],
        "delta": [],
        "gamma": [],
        "theta": [],
        "vega": [],
        "rho": [],
    }
    for _, row in df.iterrows():
        S = row.get("underlying_price")
        price = row.get("close")
        K = row.get("strike")
        expiration = row.get("expiration")
        trade_date = row.get("date")
        is_call = str(row.get("type")).lower() == "call"
        # Validate inputs
        if pd.isna([S, price, K]).any() or not expiration or not trade_date:
            iv = None
            greeks = {k: float("nan") for k in ["delta", "gamma", "theta", "vega", "rho"]}
        else:
            try:
                exp_date = dt.date.fromisoformat(str(expiration))
                tr_date = dt.date.fromisoformat(str(trade_date))
                T = (exp_date - tr_date).days / 365.0
            except ValueError:
                T = -1
            if T <= 0:
                iv = None
                greeks = {k: float("nan") for k in ["delta", "gamma", "theta", "vega", "rho"]}
            else:
                iv = implied_vol(float(price), float(S), float(K), T, rate, div_yield, is_call)
                if iv is None:
                    greeks = {k: float("nan") for k in ["delta", "gamma", "theta", "vega", "rho"]}
                else:
                    greeks = bs_greeks(float(S), float(K), T, rate, div_yield, float(iv), is_call)
        greeks_data["implied_volatility"].append(iv if iv is not None else float("nan"))
        for key in ["delta", "gamma", "theta", "vega", "rho"]:
            greeks_data[key].append(greeks[key])

    for key, values in greeks_data.items():
        df[key] = values
    return df


def download_history(
    underlying: str,
    years: int,
    max_contracts: int | None,
    contract_type: str,
    api_key: str,
    output_dir: str,
    days: int | None = None,
    start_date_str: str | None = None,
    end_date_str: str | None = None,
    rate: float = DEFAULT_RATE,
    div_yield: float = DEFAULT_DIV_YIELD,
) -> tuple[Path, int, int]:
    today = dt.date.today()
    end_date = dt.date.fromisoformat(end_date_str) if end_date_str else today
    if start_date_str:
        start_date = dt.date.fromisoformat(start_date_str)
    elif days and days > 0:
        start_date = end_date - dt.timedelta(days=days)
    else:
        start_date = end_date - dt.timedelta(days=365 * years)
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")

    contracts = fetch_option_contracts(
        underlying=underlying,
        start_date=start_date,
        end_date=end_date,
        api_key=api_key,
        max_contracts=max_contracts,
        contract_type=contract_type,
    )

    underlying_closes = fetch_underlying_closes(
        underlying=underlying,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        api_key=api_key,
    )

    all_rows: list[dict] = []
    for idx, contract in enumerate(contracts, 1):
        if idx == 1 or idx % LOG_EVERY_N_CONTRACTS == 0:
            log(f"Fetching bars for contract {idx}/{len(contracts)}: {contract['occ_symbol']}")
        bars = fetch_option_bars(
            contract,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            api_key=api_key,
            underlying_closes=underlying_closes,
        )
        all_rows.extend(bars)

    if not all_rows:
        raise RuntimeError(
            f"No option bars returned. Check ticker, date range, or API key. "
            f"Used start={start_date}, end={end_date}, contracts={len(contracts)}."
        )

    df = pd.DataFrame(all_rows)
    df.sort_values(["ticker", "date"], inplace=True)
    df = add_greeks(df, rate=rate, div_yield=div_yield)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if start_date_str or end_date_str:
        suffix = f"{start_date.isoformat()}_{end_date.isoformat()}"
    elif days:
        suffix = f"{days}d"
    else:
        suffix = f"{years}y"
    filename = output_path / f"{underlying.lower()}_options_{suffix}.csv"
    df.to_csv(filename, index=False)
    return filename, len(contracts), len(df)


def main() -> None:
    api_key = require_api_key(None)
    underlying = DEFAULT_UNDERLYING.upper()
    base_start, base_end = default_date_range()
    log(
        f"Starting download for {underlying} base range {base_start} -> {base_end} "
        f"(max_contracts={DEFAULT_MAX_CONTRACTS or 'ALL'})."
    )
    attempts = [
        (base_start, base_end),
        (
            (dt.date.fromisoformat(base_start) - dt.timedelta(days=90)).isoformat(),
            (dt.date.fromisoformat(base_end) - dt.timedelta(days=90)).isoformat(),
        ),
        (
            (dt.date.fromisoformat(base_start) - dt.timedelta(days=365)).isoformat(),
            (dt.date.fromisoformat(base_end) - dt.timedelta(days=365)).isoformat(),
        ),
    ]

    last_err: Exception | None = None
    for start_date, end_date in attempts:
        try:
            output_file, contracts, rows = download_history(
                underlying=underlying,
                years=DEFAULT_YEARS,
                days=None,
                start_date_str=start_date,
                end_date_str=end_date,
                max_contracts=DEFAULT_MAX_CONTRACTS,
                contract_type=DEFAULT_CONTRACT_TYPE,
                api_key=api_key,
                output_dir="data",
                rate=DEFAULT_RATE,
                div_yield=DEFAULT_DIV_YIELD,
            )
            print(
                f"Wrote {rows} rows for {contracts} contracts to {output_file}. "
                f"Range: {start_date} -> {end_date}. "
                "Increase --max-contracts if you want more."
            )
            return
        except Exception as err:  # pragma: no cover - runtime fallback
            last_err = err
            continue
    if last_err:
        raise last_err


if __name__ == "__main__":
    main()
