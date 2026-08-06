"""Frozen 6-asset price snapshot for the variable-weight track.

A THIRD independent snapshot. It does not read or write ``data/prices.csv``
(the Phase 1-9 input) or ``data/backtest_prices.csv`` (the Phase 8 input);
both keep their own SHA-256 and both must hash unchanged after this module
runs.

Window: 2025-07-01 to 2026-07-01, deliberately the same window as
``data/prices.csv``. The point of this track is to change the ENCODING, so
holding the estimation window fixed keeps the comparison to Phases 5-9 as
close to controlled as it can be.

ASSET SELECTION RULE, stated because it is not neutral:

  * TSLA, NVDA, AMD, COIN are RETAINED for continuity with Phases 5-9. They
    carry the selection bias documented in ``docs/backtest.md`` section 8 --
    they were chosen in 2026 with knowledge of how they performed, and no
    investor in 2021 picks that basket. Nothing about that bias is fixed by
    adding assets.
  * XOM (energy) and JNJ (healthcare) are ADDED BY SECTOR, not by return.
    The four originals correlate 0.35-0.49 with each other: they are one
    sector wearing four names, which makes the diversification term in the
    objective nearly inert. Two low-correlation, lower-volatility sectors with
    long history and no IPO-date constraint give the covariance term something
    to actually do.

The second rule is weaker than "unbiased" and should not be described as
unbiased. Choosing *sectors* rather than *tickers* by outcome removes the most
direct route for hindsight to enter, but the sectors themselves were chosen in
2026 by someone who knows which decades were kind to energy and healthcare.
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

from classical.data_pipeline import DataFetchError, TRADING_DAYS  # noqa: E402
from varweight.encoding import TICKERS                            # noqa: E402

logger = logging.getLogger("varweight.data")

DATA_DIR = PROJECT_ROOT / "data"
VW_CSV = DATA_DIR / "varweight_prices.csv"
VW_MANIFEST = DATA_DIR / "varweight_manifest.json"

#: Named only so the guard below can refuse to touch them.
PROTECTED = (DATA_DIR / "prices.csv", DATA_DIR / "backtest_prices.csv")

START_DATE = "2025-07-01"
END_DATE = "2026-07-01"
MIN_ROWS = 200


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prices(tickers=TICKERS) -> pd.DataFrame:
    """Load the frozen snapshot, verifying its recorded SHA-256."""
    tickers = list(tickers)
    if not VW_CSV.exists():
        raise DataFetchError(
            f"{VW_CSV.name} is absent. Run:\n    python varweight/data.py --fetch"
        )
    prices = pd.read_csv(VW_CSV, index_col=0, parse_dates=True)
    if list(prices.columns) != tickers:
        raise DataFetchError(
            f"Column order {list(prices.columns)} != canonical {tickers}."
        )
    if VW_MANIFEST.exists():
        recorded = json.loads(VW_MANIFEST.read_text(encoding="utf-8"))
        actual = _sha256(VW_CSV)
        if recorded.get("sha256") != actual:
            raise DataFetchError(
                f"{VW_CSV.name} changed since it was frozen.\n"
                f"  manifest: {recorded.get('sha256')}\n  actual:   {actual}"
            )
    return prices


def statistics(prices: pd.DataFrame | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Annualised ``(mu, sigma, corr)`` from simple daily returns."""
    if prices is None:
        prices = load_prices()
    simple = prices.pct_change().dropna(how="any")
    mu = simple.mean().to_numpy() * TRADING_DAYS
    sigma = simple.cov().to_numpy() * TRADING_DAYS
    corr = simple.corr().to_numpy()
    sigma = 0.5 * (sigma + sigma.T)
    corr = 0.5 * (corr + corr.T)
    return mu, sigma, corr


def fetch(tickers=TICKERS, start: str = START_DATE, end: str = END_DATE,
          refresh: bool = False) -> pd.DataFrame:
    """Fetch and freeze the 6-asset snapshot. Never invents data."""
    tickers = list(tickers)
    if VW_CSV.exists() and not refresh:
        logger.info("Snapshot already frozen; use --refresh to replace.")
        return load_prices(tickers)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataFetchError("yfinance is not installed.") from exc

    logger.info("Fetching %s from %s to %s ...", tickers, start, end)
    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                          progress=False, threads=False)
    except Exception as exc:
        raise DataFetchError(
            f"yfinance download failed ({type(exc).__name__}: {exc}). "
            "No synthetic fallback is provided."
        ) from exc

    if raw is None or len(raw) == 0:
        raise DataFetchError(f"yfinance returned no rows for {tickers}.")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise DataFetchError("No 'Close' field in the response.")
        prices = raw["Close"].copy()
    else:
        raise DataFetchError("Expected a multi-ticker response.")

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise DataFetchError(f"yfinance returned no data for {missing}.")

    # yfinance sorts alphabetically; impose canonical order (the Phase 2 bug).
    prices = prices[tickers]
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().dropna(how="any")

    if len(prices) < MIN_ROWS:
        raise DataFetchError(f"Only {len(prices)} usable rows; need >= {MIN_ROWS}.")
    if (prices <= 0).to_numpy().any():
        raise DataFetchError("Non-positive prices present; data is corrupt.")
    assert list(prices.columns) == tickers, "column order contract violated"
    assert VW_CSV not in PROTECTED, "would overwrite a protected snapshot"

    prices.to_csv(VW_CSV)
    digest = _sha256(VW_CSV)
    mu, sigma, corr = statistics(prices)

    VW_MANIFEST.write_text(json.dumps({
        "purpose": "variable-weight 6-asset track; independent of the other two snapshots",
        "file": VW_CSV.name,
        "sha256": digest,
        "tickers": tickers,
        "selection_rule": (
            "TSLA/NVDA/AMD/COIN retained for continuity with Phases 5-9 and carry "
            "the selection bias in docs/backtest.md section 8; XOM and JNJ added by "
            "SECTOR (energy, healthcare) rather than by return."
        ),
        "window": {"start": start, "end": end},
        "first_date": str(prices.index[0].date()),
        "last_date": str(prices.index[-1].date()),
        "n_rows": int(len(prices)),
        "annualised_mu": {t: float(m) for t, m in zip(tickers, mu)},
        "annualised_vol": {t: float(np.sqrt(sigma[i, i])) for i, t in enumerate(tickers)},
        "correlation": [[float(x) for x in row] for row in corr],
    }, indent=2), encoding="utf-8")

    logger.info("Froze %d rows, %s to %s", len(prices),
                prices.index[0].date(), prices.index[-1].date())
    logger.info("SHA-256: %s", digest)
    return prices


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    before = {p.name: (_sha256(p) if p.exists() else None) for p in PROTECTED}
    try:
        prices = fetch(refresh=args.refresh) if (args.fetch or args.refresh) \
            else load_prices()
    except DataFetchError as exc:
        logger.error("%s", exc)
        print("VARWEIGHT SNAPSHOT: FAILED")
        return 1

    mu, sigma, corr = statistics(prices)
    tick = list(prices.columns)

    print(f"rows        : {len(prices)}  ({prices.index[0].date()} .. "
          f"{prices.index[-1].date()})")
    print(f"tickers     : {tick}")
    print(f"sha256      : {_sha256(VW_CSV)}")
    print("\nannualised mu / vol:")
    for i, t in enumerate(tick):
        print(f"  {t:<5} mu = {mu[i]:+.4f}   vol = {np.sqrt(sigma[i, i]):.4f}")
    print("\ncorrelation:")
    print("        " + "".join(f"{t:>8}" for t in tick))
    for i, t in enumerate(tick):
        print(f"  {t:<5} " + "".join(f"{corr[i, j]:>8.3f}" for j in range(len(tick))))

    old4 = corr[:4, :4]
    off4 = old4[np.triu_indices(4, k=1)]
    new_pairs = corr[:4, 4:].ravel()
    print(f"\n  original-four pairwise corr : min {off4.min():.3f}  max {off4.max():.3f}")
    print(f"  tech-vs-added pairwise corr : min {new_pairs.min():.3f}  "
          f"max {new_pairs.max():.3f}")
    print(f"  XOM-JNJ                     : {corr[4, 5]:.3f}")

    ok = True
    for p in PROTECTED:
        if p.exists():
            now = _sha256(p)
            same = now == before[p.name]
            ok &= same
            print(f"\n{p.name}: {'unchanged' if same else '*** CHANGED ***'} "
                  f"({now[:16]}...)")
    print("\nVARWEIGHT SNAPSHOT: " + ("OK" if ok else "PROTECTED FILE MODIFIED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
