"""
Baostock A-share data pipeline.

Subcommands:

    basic    Fetch stock basic info -> ./output/stock_basic.parquet
    klines   Download daily OHLCV (unadjusted) -> ./output/stock-day-YYYYMM.parquet
    adjust   Download adjust factors -> ./output/stock_adjust_factors.parquet
    calendar Download trading calendar -> ./output/calendar_dates.parquet
    forecast Download forecast reports -> ./output/forecast_reports.parquet
    index    Download index component klines -> ./output/index_component_klines.parquet
    dataset  Compose clean dataset (adjust, standardise, filter) -> ./output/dataset.parquet
    upload   Upload ./output/*.parquet to a ModelScope repository.

Setup:
    pip install baostock pandas pyarrow modelscope   # modelscope only for upload
    export MODELSCOPE_ACCESS_TOKEN=...               # only for upload
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import baostock as bs
import numpy as np
import pandas as pd

try:
    import pyarrow  # noqa: F401
except ImportError:
    raise ImportError("pip install pyarrow")


# ── constants ────────────────────────────────────────────────────────────────

FIELDS_KLINE = (
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,"
    "tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
)

INDEX_KLINE_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"

INDEX_CODE_MAP = {"hs300": "sh.000300", "zz500": "sh.000905"}

OUTPUT_DIR = "./output"
CACHE_DIR = "./cache"
MAX_RETRIES = 3


def _today() -> str:
    """Return today's date as YYYY-MM-DD."""
    from datetime import date
    return date.today().strftime("%Y-%m-%d")


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalize_code(code: str) -> str:
    """Ensure code has dot format: sh.600000 / sz.000001."""
    if "." in code:
        return code
    code = code.lower()
    code = code.replace("sh60", "sh.60")
    code = code.replace("sz00", "sz.00")
    code = code.replace("sz30", "sz.30")
    code = code.replace("sh68", "sh.68")
    return code


def _iter_rows(rs) -> list[list[str]]:
    """Drain a baostock result set into a list of row lists."""
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return rows


def _load_stock_df(path: str = "./output/stock_basic.parquet") -> pd.DataFrame:
    """Load stock basic info from parquet/CSV or fetch fresh from baostock.

    Tries in order:
      1. path (parquet or CSV)
      2. ./output/stock_basic.parquet
      3. ./local/bs_stock_basic.csv  (legacy)
      4. Fetch from baostock → save as parquet to ./output/stock_basic.parquet
    """
    p = Path(path)

    # Try given path first
    if not p.exists():
        # Fallback: try output & local
        alt_parquet = Path("./output/stock_basic.parquet")
        alt_csv = Path("./local/bs_stock_basic.csv")
        if alt_parquet.exists():
            p = alt_parquet
        elif alt_csv.exists():
            p = alt_csv

    if p.exists():
        if p.suffix == ".parquet":
            stock_df = pd.read_parquet(p, engine="pyarrow")
        else:
            stock_df = pd.read_csv(p, dtype={"type": int, "status": int})
    else:
        bs.login()
        try:
            stock_df = fetch_stock_basic()
        finally:
            bs.logout()
        if not stock_df.empty:
            out_p = Path("./output/stock_basic.parquet")
            os.makedirs(out_p.parent, exist_ok=True)
            stock_df.to_parquet(out_p, index=False, engine="pyarrow")

    if stock_df.empty:
        raise RuntimeError("Failed to load stock basic info")

    # Keep only type=1 (stock), don't filter by status (historical delisted stocks included)
    stock_df["type"] = pd.to_numeric(stock_df["type"], errors="coerce").astype(int)
    stock_df["status"] = pd.to_numeric(stock_df["status"], errors="coerce").astype(int)
    print(stock_df)
    stock_df = stock_df[stock_df["type"] == 1]
    return stock_df


# ── low-level API wrappers ───────────────────────────────────────────────────

def fetch_calendar_dates(start_date: str, end_date: str) -> pd.DataFrame:
    """Get trading calendar dates."""
    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    if rs.error_code != "0":
        print(f"query_trade_dates failed: {rs.error_code} {rs.error_msg}")
        return pd.DataFrame()
    rows = _iter_rows(rs)
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def fetch_stock_basic(code: str = "", code_name: str = "") -> pd.DataFrame:
    """Get stock basic info (code, name, ipoDate, outDate, type, status).

    Pass empty args to get all stocks.
    """
    rs = bs.query_stock_basic(code=code, code_name=code_name)
    if rs.error_code != "0":
        print(f"query_stock_basic failed: {rs.error_code} {rs.error_msg}")
        return pd.DataFrame()
    rows = _iter_rows(rs)
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def fetch_index_stock_codes(
    index_name: str, reference_date: Optional[str] = None
) -> list[str]:
    """Get constituent stock codes for hs300 / zz500."""
    if index_name == "hs300":
        rs = bs.query_hs300_stocks(date=reference_date) if reference_date else bs.query_hs300_stocks()
    elif index_name == "zz500":
        rs = bs.query_zz500_stocks(date=reference_date) if reference_date else bs.query_zz500_stocks()
    else:
        raise ValueError(f"Unsupported index: {index_name}")

    print(f"query_{index_name}_stocks: {rs.error_code} {rs.error_msg}")
    if rs.error_code != "0":
        return []
    rows = _iter_rows(rs)
    if not rows:
        return []
    stock_df = pd.DataFrame(rows, columns=rs.fields)
    return stock_df["code"].dropna().drop_duplicates().tolist() if "code" in stock_df.columns else []


def fetch_stock_adjust_factors(
    code: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Get adjust factors for a single stock."""
    code = _normalize_code(code)
    rs = bs.query_adjust_factor(code=code, start_date=start_date, end_date=end_date)
    if rs.error_code != "0":
        print(f"{code} adjust factor failed: {rs.error_code} {rs.error_msg}")
        return pd.DataFrame()
    rows = _iter_rows(rs)
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def fetch_history_k_data(
    code: str,
    start_date: str,
    end_date: str,
    fields: str = FIELDS_KLINE,
    frequency: str = "d",
    adjustflag: str = "2",
) -> pd.DataFrame:
    """Get daily klines for a single stock."""
    code = _normalize_code(code)
    print(f"fetch_history_k_data: {code}")
    rs = bs.query_history_k_data_plus(
        code=code, fields=fields, start_date=start_date, end_date=end_date,
        frequency=frequency, adjustflag=adjustflag,
    )
    if rs.error_code != "0":
        print(f"{code} failed: {rs.error_code} {rs.error_msg}")
        return pd.DataFrame()
    rows = _iter_rows(rs)
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def fetch_index_k_data(
    index_code: str,
    start_date: str,
    end_date: str,
    fields: str = INDEX_KLINE_FIELDS,
    frequency: str = "d",
) -> pd.DataFrame:
    """Get index daily klines."""
    rs = bs.query_history_k_data_plus(
        code=index_code, fields=fields, start_date=start_date, end_date=end_date,
        frequency=frequency,
    )
    if rs.error_code != "0":
        print(f"{index_code} index kline failed: {rs.error_code} {rs.error_msg}")
        return pd.DataFrame()
    rows = _iter_rows(rs)
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def fetch_forecast_report_q(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get forecast/performance report for a single stock."""
    code = _normalize_code(code)
    rs = bs.query_forecast_report(code=code, start_date=start_date, end_date=end_date)
    if rs.error_code != "0":
        print(f"{code} forecast report failed: {rs.error_code} {rs.error_msg}")
        return pd.DataFrame()
    rows = _iter_rows(rs)
    return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame()


def fetch_stock_components(
    index_names: list[str], reference_date: Optional[str] = None
) -> list[str]:
    """Get deduplicated stock codes from index names or a stocks.txt file.

    Supports: 'hs300', 'zz500', or path to a txt file (one code per line).
    """
    all_codes: set[str] = set()
    for name in index_names:
        if name.endswith(".txt") and os.path.isfile(name):
            with open(name) as f:
                codes = [line.strip() for line in f if line.strip()]
        else:
            codes = fetch_index_stock_codes(index_name=name, reference_date=reference_date)
        if not codes:
            print(f"Retrying {name} without reference_date ...")
            codes = fetch_index_stock_codes(index_name=name)
        if not codes:
            raise RuntimeError(f"Failed to get constituents for {name}")
        all_codes.update(codes)
        print(f"  {name}: {len(codes)} codes")

    result = sorted(all_codes, reverse=True)
    print(f"Total unique codes: {len(result)}")
    return result


# ── batch download ───────────────────────────────────────────────────────────

def _worker_download_chunk(
    codes: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str,
) -> int:
    """Subprocess worker: login to baostock, download each code → cache file.

    Uses "spawn" context so each worker has an independent baostock session.
    Returns the count of newly downloaded codes.
    """
    os.makedirs(cache_dir, exist_ok=True)

    lg = bs.login()
    if lg.error_code != "0":
        print(f"[worker] baostock login failed: {lg.error_code} {lg.error_msg}", flush=True)
        return 0

    downloaded = 0
    total = len(codes)
    try:
        for i, code in enumerate(codes, start=1):
            cache_file = os.path.join(
                cache_dir,
                f"{code.replace('.', '_')}_{start_date}_{end_date}.parquet",
            )
            if os.path.exists(cache_file):
                try:
                    df = pd.read_parquet(cache_file, engine="pyarrow")
                    if not df.empty:
                        if i % 50 == 0 or i == total:
                            print(f"[worker] {i}/{total} (cache)", flush=True)
                        continue
                except Exception:
                    pass

            df = pd.DataFrame()
            for attempt in range(1, MAX_RETRIES + 1):
                rs = bs.query_history_k_data_plus(
                    code=code, fields=FIELDS_KLINE,
                    start_date=start_date, end_date=end_date,
                    frequency="d", adjustflag="3",
                )
                if rs.error_code == "0":
                    rows = _iter_rows(rs)
                    if rows:
                        df = pd.DataFrame(rows, columns=rs.fields)
                    break
                if attempt < MAX_RETRIES:
                    print(f"  {code} retry {attempt}/{MAX_RETRIES}", flush=True)
                    time.sleep(attempt)

            if not df.empty:
                df.to_parquet(cache_file, index=False, engine="pyarrow")
                downloaded += 1

            if i % 50 == 0 or i == total:
                print(f"[worker] {i}/{total} (dl={downloaded})", flush=True)
    finally:
        bs.logout()

    return downloaded


def download_all_stocks_klines(
    stocks_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    prefix: str = "",
    workers: int = 4,
) -> pd.DataFrame:
    """Download daily klines (unadjusted raw) for all codes in stocks_df.

    Uses multiprocessing (spawn context) for speed: each worker has an
    independent baostock session. Cache-first: already-downloaded codes
    are loaded from ./cache/ on next run.

    Args:
        stocks_df: DataFrame with "code" column.
        start_date: Start date.
        end_date: End date.
        prefix: Only process codes starting with this.
        workers: Number of parallel download workers (default 4).

    Returns merged DataFrame.
    """
    import multiprocessing

    os.makedirs(CACHE_DIR, exist_ok=True)

    all_codes = stocks_df["code"].dropna().drop_duplicates().tolist()
    if prefix:
        all_codes = [c for c in all_codes if c.startswith(prefix)]
    total = len(all_codes)
    label = f" (prefix={prefix})" if prefix else ""
    print(f"Total stocks to process: {total}{label}")

    # Split into chunks
    chunk_size = max(1, (total + workers - 1) // workers)
    chunks = [all_codes[i : i + chunk_size] for i in range(0, total, chunk_size)]

    print(f"Workers: {workers}, chunks: {len(chunks)}, ~{chunk_size} stocks per worker")

    # "spawn" creates fresh process — no shared baostock state
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers) as pool:
        async_results = [
            pool.apply_async(
                _worker_download_chunk,
                (chunk, start_date, end_date, CACHE_DIR),
            )
            for chunk in chunks
        ]
        pool.close()
        # Progress polling
        prev_done = 0
        while True:
            time.sleep(3)
            done = sum(1 for r in async_results if r.ready())
            if done > prev_done:
                prev_done = done
                print(f"  Workers done: {done}/{len(chunks)}")
            if done == len(chunks):
                break
        pool.join()

        dl_counts = [r.get() for r in async_results]
    print(f"  New downloads: {sum(dl_counts)} / total cached")

    # Reload all from cache
    print("Loading cache ...")
    frames: list[pd.DataFrame] = []
    for idx, code in enumerate(all_codes, start=1):
        cache_file = os.path.join(
            CACHE_DIR,
            f"{code.replace('.', '_')}_{start_date}_{end_date}.parquet",
        )
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file, engine="pyarrow")
                if not df.empty:
                    frames.append(df)
            except Exception:
                pass
        if idx % 1000 == 0 or idx == total:
            print(f"  loaded {idx}/{total} from cache")

    if not frames:
        raise RuntimeError("No kline data downloaded")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["code", "date"], keep="last")
    merged = merged.sort_values(by=["date", "code"]).reset_index(drop=True)
    print(f"Done: rows={len(merged)}, stocks={merged['code'].nunique()}")
    return merged


def _worker_adjust_chunk(
    codes: list[str],
    end_date: str,
    code_start_dates: list[tuple[str, str]],
) -> list[tuple[str, pd.DataFrame]]:
    """Subprocess worker: login to baostock, download adjust factors per code.

    Returns list of (code, DataFrame) for codes with data.
    """
    ipo_map = dict(code_start_dates)
    results: list[tuple[str, pd.DataFrame]] = []
    total = len(codes)

    lg = bs.login()
    if lg.error_code != "0":
        print(f"[worker] baostock login failed: {lg.error_code} {lg.error_msg}", flush=True)
        return results

    try:
        for i, code in enumerate(codes, start=1):
            start_date = ipo_map.get(code, "1990-01-01")
            df = pd.DataFrame()
            for attempt in range(1, MAX_RETRIES + 1):
                df = fetch_stock_adjust_factors(code=code, start_date=start_date, end_date=end_date)
                if not df.empty:
                    break
                if attempt < MAX_RETRIES:
                    time.sleep(attempt)
            if not df.empty:
                results.append((code, df))

            if i % 50 == 0 or i == total:
                print(f"[worker] {i}/{total} (got={len(results)})", flush=True)
    finally:
        bs.logout()

    return results


def download_all_adjust_factors(
    stocks_df: pd.DataFrame,
    end_date: str = "",
    workers: int = 4,
) -> pd.DataFrame:
    """Download adjust factors from each stock's IPO date to end_date.

    Uses multiprocessing (spawn context): each worker has an independent
    baostock session. Saves ./output/stock_adjust_factors.parquet.
    """
    end_date = end_date.strip() or _today()

    import multiprocessing

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ipo_map = {}
    if "ipoDate" in stocks_df.columns:
        ipo_map = dict(zip(stocks_df["code"], stocks_df["ipoDate"].fillna("1990-01-01")))
    ipo_map = {k: v for k, v in ipo_map.items() if isinstance(v, str)}

    all_codes = stocks_df["code"].dropna().drop_duplicates().tolist()
    total = len(all_codes)
    print(f"Total stocks to fetch adjust factors: {total}")

    # Split codes into chunks, each with its own (code, start_date) pairs
    chunk_size = max(1, (total + workers - 1) // workers)
    chunks = [all_codes[i : i + chunk_size] for i in range(0, total, chunk_size)]
    code_date_pairs = [(c, ipo_map.get(c, "1990-01-01")) for c in all_codes]
    pair_chunks = [code_date_pairs[i : i + chunk_size] for i in range(0, total, chunk_size)]

    print(f"Workers: {workers}, chunks: {len(chunks)}, ~{chunk_size} stocks per worker")

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers) as pool:
        async_results = [
            pool.apply_async(_worker_adjust_chunk, (chunks[i], end_date, pair_chunks[i]))
            for i in range(len(chunks))
        ]
        pool.close()

        prev_done = 0
        while True:
            time.sleep(3)
            done = sum(1 for r in async_results if r.ready())
            if done > prev_done:
                prev_done = done
                print(f"  Workers done: {done}/{len(chunks)}")
            if done == len(chunks):
                break
        pool.join()

        worker_results = [r.get() for r in async_results]

    # Collect all results
    frames: list[pd.DataFrame] = []
    for wr in worker_results:
        for _code, df in wr:
            frames.append(df)

    print(f"  Downloaded: {len(frames)} stocks with adjust factor data")

    if not frames:
        raise RuntimeError("No adjust factor data downloaded")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["code", "dividOperateDate"], keep="last")
    merged = merged.sort_values(by=["code", "dividOperateDate"]).reset_index(drop=True)

    out = os.path.join(OUTPUT_DIR, "stock_adjust_factors.parquet")
    merged.to_parquet(out, index=False, engine="pyarrow")
    print(f"Saved: {out}, rows={len(merged)}, stocks={merged['code'].nunique()}")
    return merged


def download_calendar_dates(
    start_date: str = "1990-12-19",
    end_date: str = "",
) -> pd.DataFrame:
    """Download trading calendar and save to ./output/calendar_dates.parquet."""
    end_date = end_date.strip() or _today()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bs.login()
    try:
        df = fetch_calendar_dates(start_date=start_date, end_date=end_date)
    finally:
        bs.logout()

    if df.empty:
        raise RuntimeError("Failed to fetch calendar dates")

    out = os.path.join(OUTPUT_DIR, "calendar_dates.parquet")
    df.to_parquet(out, index=False, engine="pyarrow")
    print(f"Saved: {out}, rows={len(df)}")
    return df


def download_forecast_reports(
    stocks_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download forecast reports for all codes in stocks_df.

    Saves ./output/forecast_reports.parquet.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_codes = stocks_df["code"].dropna().drop_duplicates().tolist()
    total = len(all_codes)
    print(f"Total stocks for forecast reports: {total}")

    bs.login()
    try:
        frames: list[pd.DataFrame] = []
        for idx, code in enumerate(all_codes, start=1):
            df = fetch_forecast_report_q(code=code, start_date=start_date, end_date=end_date)
            if not df.empty:
                frames.append(df)
            if idx % 200 == 0 or idx == total:
                print(f"  {idx}/{total} rows={sum(len(f) for f in frames)}")
    finally:
        bs.logout()

    if not frames:
        raise RuntimeError("No forecast data downloaded")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["code", "profitForcastExpiryDate"], keep="last")
    merged = merged.sort_values(by=["code", "profitForcastExpiryDate"]).reset_index(drop=True)

    out = os.path.join(OUTPUT_DIR, "forecast_reports.parquet")
    merged.to_parquet(out, index=False, engine="pyarrow")
    print(f"Saved: {out}, rows={len(merged)}, stocks={merged['code'].nunique()}")
    return merged


def download_index_component_klines(
    index_names: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily klines for index constituent stocks.

    Args:
        index_names: List of index names (hs300, zz500) or txt file paths.
        start_date: Start date.
        end_date: End date.

    Returns merged DataFrame, saved to ./output/index_component_klines.parquet.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    codes = fetch_stock_components(index_names, reference_date=end_date)

    bs.login()
    try:
        frames: list[pd.DataFrame] = []
        for idx, code in enumerate(codes, start=1):
            df = fetch_history_k_data(
                code=code, start_date=start_date, end_date=end_date, adjustflag="2",
            )
            if not df.empty:
                frames.append(df)
            if idx % 10 == 0 or idx == len(codes):
                print(f"  {idx}/{len(codes)} rows={sum(len(f) for f in frames)}")
    finally:
        bs.logout()

    if not frames:
        raise RuntimeError("No index component kline data")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["code", "date"], keep="last")
    merged = merged.sort_values(by=["code", "date"]).reset_index(drop=True)

    out = os.path.join(OUTPUT_DIR, "index_component_klines.parquet")
    merged.to_parquet(out, index=False, engine="pyarrow")
    print(f"Saved: {out}, rows={len(merged)}, stocks={merged['code'].nunique()}")
    return merged


# ── dataset builder ───────────────────────────────────────────────────────────

# Price columns that factor applies to (renamed snake_case: preclose → pre_close).
PRICE_COLS_ADJ = ["open", "high", "low", "close", "pre_close"]

# Column rename map: baostock raw → clean snake_case.
# Field meanings per baostock API docs:
#   date       - exchange date
#   code       - security code (sh.XXXXXX / sz.XXXXXX)
#   open/high/low/close - OHLC prices
#   preclose   - previous trading day close (as reported by exchange, adjusts for ex-dividend)
#   volume     - cumulative volume (unit: 股 / shares)
#   amount     - turnover (unit: CNY / 人民币元)
#   adjustflag - 1=back-adjusted, 2=fore-adjusted, 3=unadjusted
#   turn       - turnover rate = volume / float_shares * 100 (%)
#   tradestatus- 1=normal, 0=suspended
#   pctChg     - daily change % = (close-preclose)/preclose * 100
#   peTTM      - trailing P/E
#   pbMRQ      - P/B (MRQ)
#   psTTM      - trailing P/S
#   pcfNcfTTM  - trailing P/CF
#   isST       - 1=ST, 0=normal
_RENAME_MAP = {
    "preclose": "pre_close",
    "turn": "turnover_rate",
    "pctChg": "change_pct",
    "adjustflag": "adjust_method",
    "tradestatus": "trade_status",
    "isST": "is_st",
    "peTTM": "pe_ttm",
    "psTTM": "ps_ttm",
    "pcfNcfTTM": "pcf_ncf_ttm",
    "pbMRQ": "pb_mrq",
}


def _load_monthly_klines(
    kline_dir: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load and concat stock-day-YYYYMM.parquet files, optionally filtered by date.

    Only loads files whose YYYYMM falls within [start_date, end_date] to reduce I/O.
    """
    import re

    all_files = sorted(Path(kline_dir).glob("stock-day-*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No stock-day-*.parquet files in {kline_dir}")

    # Filter by YYYYMM extracted from filename
    files = all_files
    if start_date or end_date:
        start_ym = int(start_date[:4] + start_date[5:7]) if start_date else 0
        end_ym = int(end_date[:4] + end_date[5:7]) if end_date else 999999
        files = [
            f for f in all_files
            if (m := re.search(r"(\d{6})", f.name))
            and start_ym <= int(m.group(1)) <= end_ym
        ]

    if not files:
        raise FileNotFoundError(
            f"No stock-day-*.parquet files matched date range [{start_date}, {end_date}]"
        )
    print(f"Loading {len(files)} of {len(all_files)} monthly files from {kline_dir} ...")
    return pd.concat(
        pd.read_parquet(f, engine="pyarrow") for f in files
    ).reset_index(drop=True)


def _apply_adjust_factors(
    df: pd.DataFrame,
    adjust_df: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    """Apply adjust factors via searchsorted — robust for disjoint date ranges.

    For each code, uses numpy searchsorted to map kline dates → factor index
    in the adjust-date array.  Handles cases where kline dates are completely
    outside the adjust-date range (gaps are filled with 1.0).

    Accounting invariant: amount = price × volume (UNCHANGED).
    """
    if mode not in ("fore", "back"):
        raise ValueError(f"adjust mode must be 'fore' or 'back', got '{mode}'")

    factor_col = "foreAdjustFactor" if mode == "fore" else "backAdjustFactor"
    # fore: use NEXT ex-div factor for dates BEFORE it → searchsorted(right) - 1 + 1 = right
    #       actually: row_date < ex-div → use that factor. So for date BEFORE first ex-div,
    #       the factor from the first ex-div applies.  searchsorted(right) gives the insertion
    #       point — the first index where adj_date > row_date. That's the factor we want.
    #       For dates past the last ex-div → factor = 1.0
    # back: use CURRENT ex-div factor for dates AT/AFTER it → searchsorted(right) - 1
    #       gives the last index where adj_date ≤ row_date.

    # Build {code: (dates_array, factors_array)} for vectorized searchsorted
    adj = adjust_df[["code", "dividOperateDate", factor_col]].dropna().copy()
    adj["code"] = adj["code"].apply(lambda x: str(x).replace(".", "").upper())
    adj["date"] = pd.to_datetime(adj["dividOperateDate"])
    adj = adj.drop(columns=["dividOperateDate"])
    adj = adj.drop_duplicates(subset=["code", "date"], keep="last")

    factor_src: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for code, grp in adj.groupby("code", sort=False):
        grp = grp.sort_values("date")
        dates_i = grp["date"].values.astype("datetime64[ns]").astype("int64")
        vals_f = pd.to_numeric(grp[factor_col], errors="coerce").fillna(1.0).values.astype("float32")
        factor_src[code] = (dates_i, vals_f)

    parts: list[pd.DataFrame] = []
    n_miss = 0
    for code, code_df in df.groupby(level="code", sort=False, observed=True):
        src = factor_src.get(code)
        if src is None:
            n_miss += 1
            f = np.ones(len(code_df), dtype="float32")
        else:
            adj_dates_i, adj_vals = src
            d_i = code_df.index.get_level_values("date").values.astype("datetime64[ns]").astype("int64")

            if mode == "fore":
                # searchsorted(right) → first index where adj_date > row_date
                idx = adj_dates_i.searchsorted(d_i, side="right")
                f = np.ones(len(d_i), dtype="float32")
                mask = idx < len(adj_vals)
                f[mask] = adj_vals[idx[mask]]
            else:
                # searchsorted(right) - 1 → last index where adj_date ≤ row_date
                idx = adj_dates_i.searchsorted(d_i, side="right") - 1
                f = np.ones(len(d_i), dtype="float32")
                mask = idx >= 0
                f[mask] = adj_vals[idx[mask]]

        code_df = code_df.copy()
        for col in PRICE_COLS_ADJ:
            if col in code_df.columns:
                code_df[col] = (code_df[col].astype("float32") * f).round(4)
        if "volume" in code_df.columns:
            code_df["volume"] = (code_df["volume"].astype("float32") / f).round(2)
        code_df["adjust_factor"] = f
        parts.append(code_df)

    if n_miss:
        print(f"  {n_miss} codes had no adjust factors (factor=1.0 used)")

    df = pd.concat(parts).sort_index()
    df["adjust_method"] = mode
    return df


def normalize_klines_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise and clean a baostock kline DataFrame with minimal memory footprint.

    - Standardise code format (SH600000 / SZ000001)
    - Rename columns to snake_case
    - Cast numeric types to float32 / int8 explicitly
    - OHLC sanity: enforce high >= max(O,C,H,L), low <= min(O,C,H,L)
    - Derive: amount (元) = volume(股) × close, change, change_pct(%)
    - Filter: trade_status=0 (suspended), volume <= 0
    - Set multi-index (code, date), deduplicate, sort
    """
    if df is None or df.empty:
        return df

    df = df.reset_index(drop=False)
    df["code"] = df["code"].apply(lambda x: str(x).replace(".", "").upper())
    df.rename(columns=_RENAME_MAP, inplace=True)

    # date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # OHL → float32
    for col in ["open", "high", "low"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # close & pre_close → float32
    for col in ["close", "pre_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # ── OHLC sanity ──
    # Negative/zero prices → NaN (A-share prices always > 0)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = np.nan
    # Enforce high >= max(O,C,L) and low <= min(O,C,H)
    if "high" in df.columns and "low" in df.columns:
        price_cols = ["open", "close", "high", "low"]
        available = [c for c in price_cols if c in df.columns]
        if available:
            hi = df[available].max(axis=1)
            lo = df[available].min(axis=1)
            df["high"] = hi.astype("float32")
            df["low"] = lo.astype("float32")

    # volume → float32
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("float32")

    # ── amount: API returns CNY. volume unit = 股(shares).
    #    If already adjusted, amount is the invariant historical fact — DO NOT recompute.
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype("float32")
    elif "volume" in df.columns and "close" in df.columns:
        df["amount"] = (df["volume"].astype("float32") * df["close"].astype("float32")).astype("float32")

    # change_pct — drop API value, always recompute from close/pre_close
    # to guarantee consistency with (possibly adjusted) prices.
    if "change_pct" in df.columns:
        df = df.drop(columns=["change_pct"])

    # ── turnover_rate: only use API value, never fabricate ──
    # (fallback formula amount/volume is wrong — turnover = vol / float_shares)
    if "turnover_rate" in df.columns:
        df["turnover_rate"] = pd.to_numeric(df["turnover_rate"], errors="coerce").astype("float32")

    # ── filter before index: suspended / non-trading ──
    if "trade_status" in df.columns:
        df = df[df["trade_status"].astype(str) != "0"]

    # drop zero/negative volume rows
    df = df[df["volume"] > 0]

    # drop rows with invalid close (NaN from OHLC sanity or API error)
    df = df.dropna(subset=["close"])

    # set multi-index, sort, deduplicate
    df = df.set_index(["code", "date"], drop=True).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    g = df.groupby(level="code", observed=True)

    # ── change: recompute from (possibly adjusted) close/pre_close ──
    if "pre_close" in df.columns:
        df["change"] = (df["close"] - df["pre_close"]).astype("float32")
    else:
        df["change"] = (g["close"].shift(0) - g["close"].shift(1)).astype("float32")

    # ── change_pct: recompute (matching API convention: e.g. 2.5 = 2.5%) ──
    if "pre_close" in df.columns:
        df["change_pct"] = ((df["close"] / df["pre_close"] - 1.0) * 100.0).astype("float32")
    else:
        df["change_pct"] = (g["close"].pct_change() * 100.0).astype("float32")

    # flag columns → int8
    for col in ["is_st", "trade_status"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")

    # adjust_factor: default 1.0 (no adjustment). build_dataset overrides after adjust.
    df["adjust_factor"] = np.float32(1.0)

    # cleanup

    # valuation → float32
    for col in ["pe_ttm", "ps_ttm", "pcf_ncf_ttm", "pb_mrq"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # cleanup
    if "index" in df.columns:
        df.drop(columns=["index"], inplace=True)

    return df


def build_dataset(
    kline_input: str | pd.DataFrame = OUTPUT_DIR,
    adjust_path: str | None = "output/stock_adjust_factors.parquet",
    basic_path: str | None = "output/stock_basic.parquet",
    adjust: str = "none",
    remove_st: bool = True,
    drop_ipo_days: int = 0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Build a clean, standardised OHLCV dataset from raw output files.

    Pipeline:
        1. Load monthly files filtered by YYYYMM (reduces I/O).
        2. Date filter early to shrink data before expensive steps.
        3. normalize_klines_df (rename, cast f32/i8, OHLC sanity, filter, multi-index).
        4. Apply adjust factors on clean numeric data (merge_asof, single pass).
        5. Optionally drop first N trading days per stock after IPO.
        6. Row-level ST removal via is_st flag.

    Args:
        kline_input: Directory of stock-day-*.parquet, or a raw DataFrame.
        adjust_path: Path to adjust factors parquet.
        adjust: "none", "fore", or "back".
        remove_st: Remove rows where is_st == 1 (row-level, keeps normal data).
        drop_ipo_days: Drop first N trading days per stock (0 = keep all, e.g. 20).
        start_date: Start date filter (inclusive).
        end_date: End date filter (inclusive).

    Returns:
        Normalised DataFrame with multi-index (code, date).
    """
    # 1. Load (filter files by YYYYMM in filename to reduce I/O)
    if isinstance(kline_input, pd.DataFrame):
        df = kline_input.copy()
    else:
        df = _load_monthly_klines(kline_input, start_date=start_date, end_date=end_date)
    print(f"Loaded: rows={len(df)}, stocks={df['code'].nunique()}")

    # 2. Date filter early (before expensive adjust/normalize) on raw data
    df["date"] = pd.to_datetime(df["date"])
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    if df.empty:
        raise RuntimeError("No data after date filter")

    # 3. Normalise first (rename, cast, filter, multi-index — single cleaning pass)
    df = normalize_klines_df(df)
    n_stocks = df.index.get_level_values("code").nunique()
    print(f"Normalised: rows={len(df)}, stocks={n_stocks}")

    # 4. Adjust factors on already-numeric data
    if adjust != "none":
        if adjust_path is None or not os.path.exists(adjust_path):
            raise ValueError(f"adjust_path not found: {adjust_path}")
        adjust_df = pd.read_parquet(adjust_path, engine="pyarrow")
        print(f"Applying {adjust} ({len(adjust_df)} factor rows) ...")
        df = _apply_adjust_factors(df, adjust_df, mode=adjust)
        # Recompute derived columns from adjusted prices
        g = df.groupby(level="code", observed=True)
        if "pre_close" in df.columns:
            df["change"] = (df["close"] - df["pre_close"]).astype("float32")
            df["change_pct"] = ((df["close"] / df["pre_close"] - 1.0) * 100.0).astype("float32")
        else:
            df["change"] = (g["close"].shift(0) - g["close"].shift(1)).astype("float32")
            df["change_pct"] = (g["close"].pct_change() * 100.0).astype("float32")
    else:
        df["adjust_method"] = "none"

    # 5. Drop first N trading days per stock (IPO window removal).
    #    Uses stock_basic ipoDate: only clips stocks whose first data date is
    #    within 60 calendar days of actual IPO. Stocks appearing mid-lifecycle
    #    (e.g. data starts 2020 but IPOd 2005) are left untouched.
    if drop_ipo_days > 0:
        ipo_dates: dict[str, pd.Timestamp] = {}
        if basic_path and os.path.exists(basic_path):
            basic = pd.read_parquet(basic_path, engine="pyarrow")
            basic["code_norm"] = basic["code"].apply(
                lambda x: str(x).replace(".", "").upper()
            )
            for _, row in basic.iterrows():
                dt = pd.to_datetime(row["ipoDate"], errors="coerce")
                if pd.notna(dt):
                    ipo_dates[row["code_norm"]] = dt

        before = len(df)
        codes = df.index.get_level_values("code")
        dates = df.index.get_level_values("date")

        # Vectorised: first date per code + row-number-within-code
        first_date = dates.to_series().groupby(codes, observed=True).transform("first")
        rank = dates.to_series().groupby(codes, observed=True).rank("first").astype(int)

        # Determine which stocks are "IPO-adjacent" (first data date within 60d of IPO)
        code_first = first_date.to_frame("first_date")
        code_first["code"] = codes
        code_first["ipo_date"] = codes.map(lambda c: ipo_dates.get(c, pd.NaT))
        code_first["is_ipo"] = (
            code_first["ipo_date"].notna()
            & ((code_first["first_date"] - code_first["ipo_date"]).dt.days <= 60)
        )

        # Keep row if: rank > drop_ipo_days OR stock is not IPO-adjacent
        keep = (rank > drop_ipo_days) | (~code_first["is_ipo"])
        df = df[keep.values]

        print(f"Dropped first {drop_ipo_days} IPO-adjacent days: {before} → {len(df)} rows")

    # 6. Remove ST periods (row-level filter via is_st flag)
    if remove_st:
        if "is_st" in df.columns:
            before = len(df)
            df = df[df["is_st"] == 0]
            print(f"Removed ST rows: {before} → {len(df)} rows")
        # Also drop delisted stocks that appear only with is_st=1 throughout
        # (some data sources label delisted stocks as ST and have no normal rows)
        if "is_st" not in df.columns:
            print("  (no is_st column, skipping ST removal)")

    print(f"Dataset ready: rows={len(df)}, "
          f"stocks={df.index.get_level_values('code').nunique()}, "
          f"from {df.index.get_level_values('date').min().date()} "
          f"to {df.index.get_level_values('date').max().date()}")
    return df


# ── post-processing ──────────────────────────────────────────────────────────

def split_to_monthly_parquet(
    df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Split merged kline DataFrame into monthly parquet: stock-day-YYYYMM.parquet.

    Keeps: code, date, open, high, low, close, volume.
    """
    os.makedirs(output_dir, exist_ok=True)
    keep = ["code", "date", "open", "high", "low", "close", "volume"]
    df = df[keep].copy()
    df["ym"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m")

    for ym, group in df.groupby("ym"):
        path = os.path.join(output_dir, f"stock-day-{ym}.parquet")
        group.drop(columns=["ym"]).to_parquet(path, index=False, engine="pyarrow")
        print(f"  {path}: {len(group)} rows, {group['code'].nunique()} stocks")

    print(f"Done: {df['ym'].nunique()} monthly files -> {output_dir}")


# ── upload ───────────────────────────────────────────────────────────────────

def upload_to_modelscope(
    repo_id: str,
    repo_type: str = "dataset",
    allow_patterns: str = "*.parquet",
    local_dir: str = OUTPUT_DIR,
    path_in_repo: str = ""
) -> None:
    """Upload *.parquet from local_dir to a ModelScope repository.

    Requires MODELSCOPE_ACCESS_TOKEN env var or login via modelscope CLI.
    """
    try:
        from modelscope.hub.api import HubApi
    except ImportError:
        raise ImportError("pip install modelscope")

    token = os.environ.get("MODELSCOPE_ACCESS_TOKEN", "")
    if not token:
        raise RuntimeError("Set MODELSCOPE_ACCESS_TOKEN env var")

    files = sorted(Path(local_dir).glob("*.parquet"))
    if not files:
        print(f"No parquet files in {local_dir}")
        return

    api = HubApi()
    api.login(access_token=token)

    print(f"Uploading {len(files)} files to modelscope:{repo_id}")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_dir,
        path_in_repo=path_in_repo,
        allow_patterns=allow_patterns,
        commit_message=f"Update stock-day data ({len(files)} files)",
        repo_type=repo_type,
        token=token,
    )
    print(f"Done: https://www.modelscope.cn/{repo_type}s/{repo_id}")


# ── stock basic ──────────────────────────────────────────────────────────────

def download_stock_basic(
    output_dir: str = OUTPUT_DIR,
) -> pd.DataFrame:
    """Fetch all stock basic info and save to ./output/stock_basic.parquet."""
    os.makedirs(output_dir, exist_ok=True)
    bs.login()
    try:
        df = fetch_stock_basic()
    finally:
        bs.logout()

    if df.empty:
        raise RuntimeError("Failed to fetch stock basic info")

    out = os.path.join(output_dir, "stock_basic.parquet")
    df.to_parquet(out, index=False, engine="pyarrow")
    print(f"Saved: {out}, rows={len(df)}, stocks={df['code'].nunique()}")
    return df


# ── helpers ──────────────────────────────────────────────────────────────────

def _progress(idx: int, total: int, frames: list) -> None:
    """Print progress at checkpoints."""
    if idx % 100 == 0 or idx == total:
        print(f"  {idx}/{total} rows={sum(len(f) for f in frames)}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cmd_basic(args: argparse.Namespace) -> None:
    """Fetch stock basic info and save to output."""
    download_stock_basic()

    if args.upload:
        _upload_check(args)
        upload_to_modelscope(repo_id=args.repo_id, repo_type="dataset", allow_patterns="stock_basic.parquet")


def _cmd_klines(args: argparse.Namespace) -> None:
    """Download A-share daily klines -> split to monthly parquet."""
    stock_df = _load_stock_df()
    print(f"Stocks to process: {len(stock_df)}")

    df = download_all_stocks_klines(
        stocks_df=stock_df,
        start_date=args.start_date,
        end_date=args.end_date,
        prefix=args.prefix,
        workers=args.workers,
    )
    split_to_monthly_parquet(df)

    if args.upload:
        _upload_check(args)
        upload_to_modelscope(repo_id=args.repo_id, repo_type="dataset", allow_patterns="stock-day-*.parquet", path_in_repo="daily/")


def _cmd_adjust(args: argparse.Namespace) -> None:
    """Download adjust factors for all stocks."""
    stock_df = _load_stock_df()
    print(f"Stocks to process: {len(stock_df)}")

    download_all_adjust_factors(stocks_df=stock_df, end_date=args.end_date, workers=args.workers)

    if args.upload:
        _upload_check(args)
        upload_to_modelscope(repo_id=args.repo_id, repo_type="dataset", allow_patterns="stock-adjust-*.parquet", path_in_repo="adjust/")


def _cmd_calendar(args: argparse.Namespace) -> None:
    """Download trading calendar dates."""
    download_calendar_dates(start_date=args.start_date, end_date=args.end_date)

    if args.upload:
        _upload_check(args)
        upload_to_modelscope(repo_id=args.repo_id, repo_type="dataset", allow_patterns="stock-calendar-*.parquet", path_in_repo="calendar/")


def _cmd_forecast(args: argparse.Namespace) -> None:
    """Download forecast reports for all stocks."""
    stock_df = _load_stock_df()
    print(f"Stocks to process: {len(stock_df)}")

    download_forecast_reports(
        stocks_df=stock_df,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if args.upload:
        _upload_check(args)
        upload_to_modelscope(repo_id=args.repo_id, repo_type="dataset", allow_patterns="stock-forecast-*.parquet", path_in_repo="forecast/")


def _cmd_index(args: argparse.Namespace) -> None:
    """Download index component klines."""
    index_names = args.index_scope.split(",")
    print(f"Index scope: {index_names}")

    download_index_component_klines(
        index_names=index_names,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if args.upload:
        _upload_check(args)
        upload_to_modelscope(repo_id=args.repo_id, repo_type="dataset", allow_patterns="stock-index-*.parquet", path_in_repo="index/")


def _cmd_dataset(args: argparse.Namespace) -> None:
    """Build clean dataset from output files."""
    df = build_dataset(
        kline_input=args.kline_dir,
        adjust_path=args.adjust_path,
        basic_path=args.basic_path,
        adjust=args.adjust,
        remove_st=not args.keep_st,
        drop_ipo_days=args.drop_ipo_days,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    out = os.path.join(OUTPUT_DIR, "stock_ohlcv_d.parquet")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.reset_index().to_parquet(out, index=False, engine="pyarrow")
    print(f"Saved: {out}")

    if args.upload:
        _upload_check(args)
        upload_to_modelscope(repo_id=args.repo_id, repo_type="dataset", allow_patterns="stock_ohlcv_d.parquet", path_in_repo="")


def _cmd_upload(args: argparse.Namespace) -> None:
    """Upload existing output files to ModelScope."""
    upload_to_modelscope(local_dir=args.local_dir, repo_id=args.repo_id)


def _upload_check(args: argparse.Namespace) -> None:
    if not args.repo_id:
        raise SystemExit("--repo-id is required when --upload is set")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baostock A-share data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Subcommands")
    sub.required = True

    # basic
    p_basic = sub.add_parser("basic", help="Fetch stock basic info (code, name, ipoDate, type, status)")
    p_basic.add_argument("--upload", action="store_true", help="Upload to ModelScope")
    p_basic.add_argument("--repo-id", default="", help="ModelScope repo (required if --upload)")
    p_basic.set_defaults(func=_cmd_basic)

    # klines
    p_klines = sub.add_parser("klines", help="Download daily klines (raw, unadjusted)")
    p_klines.add_argument("--start-date", default="2000-01-01", help="Start date (default: 2000-01-01)")
    p_klines.add_argument("--end-date", default=_today(), help="End date (default: today) (default: 2026-12-31)")
    p_klines.add_argument("--prefix", default="", help="Only codes starting with prefix (e.g. sz.)")
    p_klines.add_argument("--workers", type=int, default=4, help="Parallel download workers (default: 4)")
    p_klines.add_argument("--upload", action="store_true", help="Upload to ModelScope after split")
    p_klines.add_argument("--repo-id", default="", help="ModelScope repo (required if --upload)")
    p_klines.set_defaults(func=_cmd_klines)

    # adjust
    p_adj = sub.add_parser("adjust", help="Download adjust factors")
    p_adj.add_argument("--end-date", default=_today(), help="End date (default: today) (default: 2026-12-31)")
    p_adj.add_argument("--prefix", default="", help="Only codes starting with prefix (e.g. sz.)")
    p_adj.add_argument("--workers", type=int, default=4, help="Parallel download workers (default: 4)")
    p_adj.add_argument("--upload", action="store_true", help="Upload to ModelScope after processing")
    p_adj.add_argument("--repo-id", default="", help="ModelScope repo (required if --upload)")
    p_adj.set_defaults(func=_cmd_adjust)

    # calendar
    p_cal = sub.add_parser("calendar", help="Download trading calendar dates")
    p_cal.add_argument("--start-date", default="1990-12-19", help="Start date")
    p_cal.add_argument("--end-date", default=_today(), help="End date (default: today)")
    p_cal.add_argument("--upload", action="store_true", help="Upload to ModelScope")
    p_cal.add_argument("--repo-id", default="", help="ModelScope repo (required if --upload)")
    p_cal.set_defaults(func=_cmd_calendar)

    # forecast
    p_fc = sub.add_parser("forecast", help="Download forecast reports")
    p_fc.add_argument("--start-date", default="2025-01-01", help="Start date")
    p_fc.add_argument("--end-date", default=_today(), help="End date (default: today)")
    p_fc.add_argument("--upload", action="store_true", help="Upload to ModelScope")
    p_fc.add_argument("--repo-id", default="", help="ModelScope repo (required if --upload)")
    p_fc.set_defaults(func=_cmd_forecast)

    # index
    p_idx = sub.add_parser("index", help="Download index component klines (front-adjusted)")
    p_idx.add_argument(
        "--index-scope", default="hs300",
        help="Comma-separated index names or txt paths (e.g. hs300,zz500 or stocks.txt)",
    )
    p_idx.add_argument("--start-date", default="2020-01-01", help="Start date")
    p_idx.add_argument("--end-date", default=_today(), help="End date (default: today)")
    p_idx.add_argument("--upload", action="store_true", help="Upload to ModelScope")
    p_idx.add_argument("--repo-id", default="", help="ModelScope repo (required if --upload)")
    p_idx.set_defaults(func=_cmd_index)

    # dataset
    p_bld = sub.add_parser("dataset", help="Build clean standardised dataset from output files")
    p_bld.add_argument("--kline-dir", default=OUTPUT_DIR, help="Directory of stock-day-*.parquet files")
    p_bld.add_argument("--adjust-path", default="output/stock_adjust_factors.parquet", help="Adjust factors parquet")
    p_bld.add_argument("--basic-path", default="output/stock_basic.parquet", help="Stock basic parquet (for IPO date)")
    p_bld.add_argument("--adjust", default="none", choices=["none", "fore", "back"], help="Adjustment mode")
    p_bld.add_argument("--keep-st", action="store_true", help="Keep ST stocks (default: remove)")
    p_bld.add_argument("--drop-ipo-days", type=int, default=20, help="Drop first N trading days per stock (default: 20)")
    p_bld.add_argument("--start-date", default=None, help="Filter start date")
    p_bld.add_argument("--end-date", default=None, help="Filter end date")
    p_bld.add_argument("--upload", action="store_true", help="Upload to ModelScope after build")
    p_bld.add_argument("--repo-id", default="", help="ModelScope repo (required if --upload)")
    p_bld.set_defaults(func=_cmd_dataset)

    # upload
    p_up = sub.add_parser("upload", help="Upload existing output to ModelScope")
    p_up.add_argument("--repo-id", required=True, help="ModelScope repo (e.g. yourname/stock-data)")
    p_up.add_argument("--local-dir", default=OUTPUT_DIR, help="Local directory to upload")
    p_up.set_defaults(func=_cmd_upload)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
