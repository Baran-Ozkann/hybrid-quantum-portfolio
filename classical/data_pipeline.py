"""
classical/data_pipeline.py -- Phase 2: market data and portfolio statistics.

Fetches one year of daily closing prices for four volatile equities, converts
them to annualised expected returns and a 4x4 covariance matrix, and persists
everything to ``data/`` for the later phases.

ASSET ORDER IS LOAD-BEARING
---------------------------
Every downstream array -- expected returns, covariance, Ising coefficients,
qubit indices, portfolio weights -- is positional. ``TICKERS`` defines the
canonical order and qubit *i* represents ``TICKERS[i]``.

``yfinance`` does NOT preserve the requested order: it returns columns sorted
alphabetically (AMD, COIN, NVDA, TSLA). Silently zipping those values against
``TICKERS`` would attach every asset's statistics to the wrong name and the
error would be invisible in the output -- the numbers all look plausible. The
fetch therefore reindexes explicitly to ``TICKERS`` and asserts the result.

RETURN CONVENTION
-----------------
Simple (arithmetic) returns, ``r_t = P_t / P_{t-1} - 1``, annualised as
``mu = 252 * mean(r)`` and ``Sigma = 252 * cov(r)``. This is the textbook
Markowitz convention: portfolio return is linear in simple returns, so
``w' mu`` is the portfolio's expected return. (Log returns aggregate better
across TIME but not across ASSETS -- the log return of a portfolio is not the
weighted mean of log returns -- so they are the wrong input for mean-variance
optimisation. Daily log returns are still saved for reference.)

NETWORK POLICY
--------------
If the fetch fails or returns unusable data this module raises
``DataFetchError``. It never falls back to synthetic prices: a silent
substitution would make every downstream result meaningless while still
appearing to succeed.

Usage::

    python classical/data_pipeline.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("data_pipeline")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Canonical asset order. Qubit i <-> TICKERS[i] throughout the project.
TICKERS: tuple[str, ...] = ("TSLA", "NVDA", "AMD", "COIN")

#: Fixed historical window (both endpoints in the past) so every run of this
#: project sees identical data. Required for the Phase 7 reproducibility runs.
START_DATE = "2025-07-01"
END_DATE = "2026-07-01"

#: NYSE trading days per year, the standard annualisation factor.
TRADING_DAYS = 252

#: Annualised risk-free rate used for Sharpe ratios (~3-month T-bill).
RISK_FREE_RATE = 0.05


class DataFetchError(RuntimeError):
    """Raised when market data cannot be retrieved or fails sanity checks."""


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

PRICES_CSV = DATA_DIR / "prices.csv"


def _load_cached_prices(tickers: Sequence[str]) -> pd.DataFrame | None:
    """Return the frozen price snapshot if it exists and matches ``tickers``."""
    if not PRICES_CSV.exists():
        return None
    try:
        cached = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    except Exception as exc:
        logger.warning("Could not read %s (%s); refetching.", PRICES_CSV.name, exc)
        return None
    if list(cached.columns) != list(tickers):
        logger.warning(
            "Cached prices have columns %s, expected %s; refetching.",
            list(cached.columns), list(tickers),
        )
        return None
    return cached


def fetch_price_data(
    tickers: Sequence[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return adjusted daily closing prices, in canonical ``tickers`` order.

    REPRODUCIBILITY: prefers the frozen snapshot at ``data/prices.csv`` and only
    hits the network when that file is absent or ``refresh=True``.

    This is not merely an optimisation. Yahoo does NOT return byte-identical
    data for a fixed, entirely historical window: two consecutive fetches of
    2025-07-01..2026-07-01 were measured to differ, with NVDA's annualised mean
    moving by 1.0e-9 and its covariance entries by up to 3.7e-8 (all other
    tickers unchanged). The cause is vendor-side revision of split/dividend
    adjustment factors. The magnitude is financially irrelevant, but it
    propagates into the Ising coefficients, so a re-fetch silently changes the
    problem the saved parameters were trained against. Freezing the snapshot
    makes every downstream artifact exactly reproducible.

    Args:
        tickers: Symbols to fetch, in the order the caller wants them.
        start: Inclusive start date, ``YYYY-MM-DD``.
        end: Exclusive end date, ``YYYY-MM-DD``.
        refresh: Force a network fetch, overwriting the frozen snapshot.

    Returns:
        DataFrame indexed by date whose columns are exactly ``tickers`` in the
        given order, with no missing values.

    Raises:
        DataFetchError: on network failure, empty response, unexpected schema,
            missing tickers, or too few usable observations.
    """
    tickers = list(tickers)

    if not refresh:
        cached = _load_cached_prices(tickers)
        if cached is not None:
            logger.info(
                "Using frozen price snapshot %s (%d rows, %s to %s). "
                "Pass refresh=True to refetch.",
                PRICES_CSV.name, len(cached),
                cached.index[0].date(), cached.index[-1].date(),
            )
            return cached

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment guard
        raise DataFetchError(
            "yfinance is not installed. Run: pip install -r requirements.txt"
        ) from exc

    logger.info("Fetching %s from %s to %s over the network ...", tickers, start, end)

    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,   # adjust for splits/dividends
            progress=False,
            threads=False,      # deterministic ordering, clearer errors
        )
    except Exception as exc:
        raise DataFetchError(
            f"yfinance download failed ({type(exc).__name__}: {exc}). "
            "Check network connectivity; no synthetic fallback is provided."
        ) from exc

    if raw is None or len(raw) == 0:
        raise DataFetchError(
            f"yfinance returned no rows for {tickers} between {start} and {end}."
        )

    # A multi-ticker request yields MultiIndex columns (field, ticker); a
    # single-ticker request yields flat columns.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise DataFetchError(
                f"No 'Close' field in response; got {sorted(set(raw.columns.get_level_values(0)))}."
            )
        prices = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise DataFetchError(f"No 'Close' column in response; got {list(raw.columns)}.")
        prices = raw[["Close"]].copy()
        prices.columns = tickers[:1]

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise DataFetchError(
            f"yfinance returned no data for {missing}. "
            f"Available: {list(prices.columns)}."
        )

    # THE CRITICAL STEP: impose canonical order. yfinance sorts alphabetically.
    prices = prices[tickers]
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    before = len(prices)
    prices = prices.dropna(how="any")
    if before != len(prices):
        logger.warning("Dropped %d rows containing missing prices.", before - len(prices))

    if len(prices) < 100:
        raise DataFetchError(
            f"Only {len(prices)} usable trading days returned; expected ~250 for a "
            "one-year window. Refusing to build statistics from so little data."
        )
    if (prices <= 0).to_numpy().any():
        raise DataFetchError("Non-positive prices present; data is corrupt.")

    # Contract check -- cheap, and the failure mode it guards is silent.
    assert list(prices.columns) == tickers, (
        f"Column order {list(prices.columns)} != canonical {tickers}"
    )

    logger.info(
        "Fetched %d trading days, %s to %s.",
        len(prices), prices.index[0].date(), prices.index[-1].date(),
    )
    logger.info("Column order verified as canonical: %s", list(prices.columns))

    # Freeze the snapshot immediately, so every later run is reproducible.
    prices.to_csv(PRICES_CSV)
    logger.info("Froze price snapshot to %s", PRICES_CSV)
    return prices


def prices_fingerprint() -> str:
    """SHA-256 of the frozen price snapshot, for provenance in statistics.json."""
    if not PRICES_CSV.exists():
        return "absent"
    return hashlib.sha256(PRICES_CSV.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def compute_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute daily simple and log returns.

    Returns:
        ``(simple_returns, log_returns)``, both with the first row dropped.
    """
    simple = prices.pct_change().dropna(how="any")
    log_r = np.log(prices / prices.shift(1)).dropna(how="any")
    logger.info("Computed returns over %d days x %d assets.", *simple.shape)
    return simple, log_r


def compute_statistics(
    simple_returns: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Annualise mean returns, covariance, and correlation.

    Args:
        simple_returns: Daily simple returns, columns in canonical order.

    Returns:
        ``(mu, sigma, corr)`` where ``mu`` has shape (n,), ``sigma`` and
        ``corr`` shape (n, n), all positionally aligned to the input columns.
    """
    mu = simple_returns.mean().to_numpy() * TRADING_DAYS
    sigma = simple_returns.cov().to_numpy() * TRADING_DAYS
    corr = simple_returns.corr().to_numpy()

    # Enforce exact symmetry: pandas' pairwise covariance can leave asymmetries
    # at the 1e-18 level, which upsets strict PSD checks downstream.
    sigma = 0.5 * (sigma + sigma.T)
    corr = 0.5 * (corr + corr.T)

    names = list(simple_returns.columns)
    logger.info("Annualised expected returns:")
    for i, name in enumerate(names):
        logger.info("    %-5s mu = %+.4f  (%+.2f%%)", name, mu[i], 100 * mu[i])

    logger.info("Annualised volatilities:")
    for i, name in enumerate(names):
        logger.info("    %-5s vol = %.4f  (%.2f%%)", name, np.sqrt(sigma[i, i]), 100 * np.sqrt(sigma[i, i]))

    logger.info("Annualised covariance matrix:")
    for i, name in enumerate(names):
        logger.info("    %-5s [%s]", name, "  ".join(f"{v:+.6f}" for v in sigma[i]))

    logger.info("Correlation matrix:")
    for i, name in enumerate(names):
        logger.info("    %-5s [%s]", name, "  ".join(f"{v:+.4f}" for v in corr[i]))

    return mu, sigma, corr


# --------------------------------------------------------------------------
# Portfolio metrics
# --------------------------------------------------------------------------

def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """Expected annualised portfolio return, ``w' mu``."""
    return float(np.dot(weights, mu))


def portfolio_volatility(weights: np.ndarray, sigma: np.ndarray) -> float:
    """Annualised portfolio volatility, ``sqrt(w' Sigma w)``."""
    var = float(weights @ sigma @ weights)
    return float(np.sqrt(max(var, 0.0)))


def sharpe_ratio(
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_free: float = RISK_FREE_RATE,
) -> float:
    """Annualised Sharpe ratio, ``(w'mu - rf) / sqrt(w' Sigma w)``."""
    vol = portfolio_volatility(weights, sigma)
    if vol < 1e-12:
        return 0.0
    return (portfolio_return(weights, mu) - risk_free) / vol


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_data(
    prices: pd.DataFrame,
    simple_returns: pd.DataFrame,
    log_returns: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
) -> dict:
    """Write prices, returns, and statistics to ``data/``.

    Returns:
        The statistics dictionary that was serialised to ``statistics.json``.
    """
    prices.to_csv(PRICES_CSV)
    simple_returns.to_csv(DATA_DIR / "daily_returns.csv")
    log_returns.to_csv(DATA_DIR / "daily_log_returns.csv")

    stats = {
        "tickers": list(prices.columns),   # from the frame, not the constant
        # Ties these statistics to the exact price snapshot they came from, so
        # a silently-changed input is detectable rather than invisible.
        "prices_sha256": prices_fingerprint(),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "first_observation": str(prices.index[0].date()),
        "last_observation": str(prices.index[-1].date()),
        "n_observations": int(len(simple_returns)),
        "trading_days_per_year": TRADING_DAYS,
        "risk_free_rate": RISK_FREE_RATE,
        "return_convention": "simple (arithmetic), annualised by *252",
        "expected_returns": mu.tolist(),
        "volatilities": np.sqrt(np.diag(sigma)).tolist(),
        "cov_matrix": sigma.tolist(),
        "correlation_matrix": corr.tolist(),
    }
    (DATA_DIR / "statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Wrote prices.csv, daily_returns.csv, statistics.json to %s", DATA_DIR)
    return stats


def load_statistics() -> dict:
    """Load ``data/statistics.json``, restoring numpy arrays.

    Raises:
        FileNotFoundError: if the pipeline has not been run.
    """
    path = DATA_DIR / "statistics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python classical/data_pipeline.py"
        )
    stats = json.loads(path.read_text(encoding="utf-8"))
    for key in ("expected_returns", "cov_matrix", "correlation_matrix", "volatilities"):
        stats[key] = np.asarray(stats[key], dtype=float)
    return stats


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_statistics(mu: np.ndarray, sigma: np.ndarray, corr: np.ndarray) -> bool:
    """Check the statistics satisfy the properties later phases rely on."""
    ok = True

    def check(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        logger.info("    [%s] %s: %s", "PASS" if passed else "FAIL", name, detail)
        if not passed:
            ok = False

    n = len(mu)
    check("shapes", sigma.shape == (n, n) and corr.shape == (n, n),
          f"mu{mu.shape}, sigma{sigma.shape}, corr{corr.shape}")
    check("finite", bool(np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))),
          "no NaN/Inf in mu or sigma")

    asym = float(np.max(np.abs(sigma - sigma.T)))
    check("symmetry", asym < 1e-12, f"max|S - S'| = {asym:.2e}")

    eig = np.linalg.eigvalsh(sigma)
    # A sample covariance from 250 observations of 4 assets is PSD by
    # construction; a negative eigenvalue would mean the matrix was corrupted.
    check("positive semi-definite", bool(np.all(eig >= -1e-10)),
          f"eigenvalues = [{', '.join(f'{v:.6f}' for v in eig)}]")

    diag_ok = bool(np.all(np.diag(sigma) > 0))
    check("positive variances", diag_ok, f"diag = {np.round(np.diag(sigma), 6).tolist()}")

    corr_ok = bool(np.allclose(np.diag(corr), 1.0, atol=1e-9)
                   and np.all(corr >= -1.0 - 1e-9) and np.all(corr <= 1.0 + 1e-9))
    check("correlation range", corr_ok, "unit diagonal, all entries within [-1, 1]")

    # Correlation must be reproducible from the covariance matrix.
    d = np.sqrt(np.diag(sigma))
    implied = sigma / np.outer(d, d)
    consistent = float(np.max(np.abs(implied - corr)))
    check("corr consistent with cov", consistent < 1e-9, f"max deviation = {consistent:.2e}")

    return ok


# --------------------------------------------------------------------------

def run_pipeline(refresh: bool = False) -> dict:
    """Fetch (or reuse the frozen snapshot), compute, validate, and persist.

    Args:
        refresh: Force a network refetch, replacing the frozen snapshot.
    """
    logger.info("=" * 64)
    logger.info("PHASE 2A -- market data pipeline")
    logger.info("=" * 64)

    prices = fetch_price_data(refresh=refresh)
    simple_returns, log_returns = compute_returns(prices)
    mu, sigma, corr = compute_statistics(simple_returns)

    logger.info("Validation:")
    if not validate_statistics(mu, sigma, corr):
        raise DataFetchError("Statistics failed validation; see log above.")

    stats = save_data(prices, simple_returns, log_returns, mu, sigma, corr)
    logger.info("Data pipeline complete.")
    return stats


if __name__ == "__main__":
    import argparse

    sys.path.insert(0, str(PROJECT_ROOT))
    from quantum.q_bridge import configure_logging

    parser = argparse.ArgumentParser(description="Phase 2 market data pipeline")
    parser.add_argument(
        "--refresh", action="store_true",
        help="refetch from the network, replacing the frozen data/prices.csv "
             "snapshot (downstream results will shift slightly; see docstring)",
    )
    args = parser.parse_args()

    configure_logging("data_pipeline.log")
    try:
        run_pipeline(refresh=args.refresh)
    except DataFetchError as exc:
        logger.error("DATA FETCH FAILED: %s", exc)
        sys.exit(1)
