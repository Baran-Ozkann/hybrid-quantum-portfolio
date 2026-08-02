"""Fetch and freeze the Phase 8 backtest price snapshot.

This is a SECOND, INDEPENDENT snapshot. It does not read, write, or otherwise
touch ``data/prices.csv``, which invariant 1 designates a frozen input to every
trained parameter set in the repository. Mixing the two would silently change
the problem that Phases 5-9 were measured against.

The two snapshots differ in purpose:

* ``data/prices.csv``          -- 1 year (2025-07..2026-07), one Hamiltonian,
                                  the instance the optimiser is judged on.
* ``data/backtest_prices.csv`` -- ~5 years (2021-06..2026-07), re-estimated in
                                  a rolling window, one Hamiltonian per fold.

COIN listed on 2021-04-14, so no earlier start is possible for this basket
without dropping an asset. The start is set to 2021-06-01 to clear the
post-IPO settling period.

Like the Phase 2 pipeline, this refuses to invent data: a failed fetch raises
rather than falling back to anything synthetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classical.data_pipeline import DataFetchError, TICKERS  # noqa: E402

logger = logging.getLogger("fetch_snapshot")

DATA_DIR = PROJECT_ROOT / "data"

#: The Phase 8 snapshot. Deliberately NOT ``prices.csv``.
BACKTEST_CSV = DATA_DIR / "backtest_prices.csv"
BACKTEST_MANIFEST = DATA_DIR / "backtest_manifest.json"

#: The Phase 1-9 snapshot, named here only so the guard below can refuse to
#: touch it.
PROTECTED_CSV = DATA_DIR / "prices.csv"

START_DATE = "2021-06-01"   # COIN IPO 2021-04-14, plus a settling margin
END_DATE = "2026-07-01"     # matches the Phase 2 snapshot's end

#: Below this many rows the rolling design (250 train + 63 test, step 63) does
#: not yield the ~16 folds the study is powered for.
MIN_ROWS = 1000


def _sha256(path: Path) -> str:
    """SHA-256 of a file, read in binary so line endings cannot drift it."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_backtest_prices(tickers=TICKERS) -> pd.DataFrame:
    """Load the frozen backtest snapshot, verifying its recorded SHA-256.

    Raises:
        DataFetchError: if the snapshot is absent, mis-ordered, or its hash
            does not match the manifest.
    """
    tickers = list(tickers)
    if not BACKTEST_CSV.exists():
        raise DataFetchError(
            f"{BACKTEST_CSV.name} is absent. Run:\n"
            f"    python backtest/fetch_snapshot.py --fetch"
        )

    prices = pd.read_csv(BACKTEST_CSV, index_col=0, parse_dates=True)
    if list(prices.columns) != tickers:
        raise DataFetchError(
            f"Column order {list(prices.columns)} != canonical {tickers}."
        )

    if BACKTEST_MANIFEST.exists():
        recorded = json.loads(BACKTEST_MANIFEST.read_text(encoding="utf-8"))
        actual = _sha256(BACKTEST_CSV)
        if recorded.get("sha256") != actual:
            raise DataFetchError(
                f"{BACKTEST_CSV.name} has changed since it was frozen.\n"
                f"  manifest: {recorded.get('sha256')}\n"
                f"  actual:   {actual}\n"
                "Every Phase 8 number was computed against the manifest hash."
            )
    return prices


def fetch(tickers=TICKERS, start: str = START_DATE, end: str = END_DATE,
          refresh: bool = False) -> pd.DataFrame:
    """Fetch ~5 years of adjusted closes and freeze them with a SHA-256.

    Args:
        tickers: Symbols, in the order the caller wants the columns.
        start: Inclusive start date.
        end: Exclusive end date.
        refresh: Overwrite an existing snapshot. Every Phase 8 figure is tied
            to the manifest hash, so this invalidates the whole phase.

    Raises:
        DataFetchError: on network failure, empty or malformed response,
            missing tickers, or too few usable rows. Never falls back to
            synthetic data.
    """
    tickers = list(tickers)

    if BACKTEST_CSV.exists() and not refresh:
        logger.info("Snapshot already frozen at %s; use --refresh to replace.",
                    BACKTEST_CSV.name)
        return load_backtest_prices(tickers)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataFetchError(
            "yfinance is not installed. Run: pip install -r requirements.txt"
        ) from exc

    logger.info("Fetching %s from %s to %s ...", tickers, start, end)
    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                          progress=False, threads=False)
    except Exception as exc:
        raise DataFetchError(
            f"yfinance download failed ({type(exc).__name__}: {exc}). "
            "Check network connectivity; no synthetic fallback is provided."
        ) from exc

    if raw is None or len(raw) == 0:
        raise DataFetchError(f"yfinance returned no rows for {tickers}.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise DataFetchError("No 'Close' field in the response.")
        prices = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise DataFetchError("No 'Close' column in the response.")
        prices = raw[["Close"]].copy()
        prices.columns = tickers[:1]

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise DataFetchError(
            f"yfinance returned no data for {missing}. "
            f"Available: {list(prices.columns)}."
        )

    # yfinance sorts alphabetically; impose canonical order. This is the Phase 2
    # bug, and it is silent -- every downstream number is wrong but plausible.
    prices = prices[tickers]
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    before = len(prices)
    prices = prices.dropna(how="any")
    if before != len(prices):
        logger.warning("Dropped %d rows with missing prices.", before - len(prices))

    if len(prices) < MIN_ROWS:
        raise DataFetchError(
            f"Only {len(prices)} usable trading days; need >= {MIN_ROWS} for the "
            "rolling design. Refusing to build a backtest from so little data."
        )
    if (prices <= 0).to_numpy().any():
        raise DataFetchError("Non-positive prices present; data is corrupt.")

    assert list(prices.columns) == tickers, "column order contract violated"
    assert BACKTEST_CSV != PROTECTED_CSV, "would overwrite the Phase 1-9 snapshot"

    prices.to_csv(BACKTEST_CSV)
    digest = _sha256(BACKTEST_CSV)

    returns = prices.pct_change().dropna(how="any")
    manifest = {
        "purpose": "Phase 8 out-of-sample backtest; independent of data/prices.csv",
        "file": BACKTEST_CSV.name,
        "sha256": digest,
        "tickers": tickers,
        "requested_start": start,
        "requested_end": end,
        "first_date": str(prices.index[0].date()),
        "last_date": str(prices.index[-1].date()),
        "n_rows": int(len(prices)),
        "n_return_rows": int(len(returns)),
        "auto_adjust": True,
        "note": (
            "Adjusted closes are point-in-time contaminated: today's split and "
            "dividend factors were not knowable during the window. Acceptable "
            "for a total-return study, but see docs/backtest.md."
        ),
    }
    BACKTEST_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Froze %d rows, %s to %s", len(prices),
                prices.index[0].date(), prices.index[-1].date())
    logger.info("SHA-256: %s", digest)
    return prices


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true",
                    help="fetch over the network if no snapshot exists")
    ap.add_argument("--refresh", action="store_true",
                    help="replace an existing snapshot (invalidates Phase 8)")
    args = ap.parse_args()

    try:
        if args.fetch or args.refresh:
            prices = fetch(refresh=args.refresh)
        else:
            prices = load_backtest_prices()
    except DataFetchError as exc:
        logger.error("%s", exc)
        print("BACKTEST SNAPSHOT: FAILED")
        return 1

    # The Phase 1-9 input must be untouched by anything in this module.
    if PROTECTED_CSV.exists():
        print(f"data/prices.csv sha256 (unchanged): {_sha256(PROTECTED_CSV)[:16]}...")

    n = len(prices)
    n_folds = max(0, (n - 1 - 250 - 63) // 63 + 1)
    print(f"rows            : {n}  ({prices.index[0].date()} .. {prices.index[-1].date()})")
    print(f"tickers         : {list(prices.columns)}")
    print(f"sha256          : {_sha256(BACKTEST_CSV)}")
    print(f"implied folds   : {n_folds}  (250 train / 63 test, step 63)")
    print("BACKTEST SNAPSHOT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
