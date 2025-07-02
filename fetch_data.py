#!/usr/bin/env python3
"""Fetch 1‑minute OHLCV bars from Polygon.io and save to CSV."""

import argparse, os
from pathlib import Path
from datetime import datetime
import requests, pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm, trange

load_dotenv()
API_KEY = os.getenv("POLY_API_KEY")
if not API_KEY:
    raise SystemExit("Set POLY_API_KEY in environment or .env file")

BASE = "https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/minute/{start}/{end}?adjusted=true&limit=50000&sort=asc&apiKey={key}"

def fetch(sym: str, start: str, end: str) -> pd.DataFrame:
    url = BASE.format(sym=sym.upper(), start=start, end=end, key=API_KEY)
    js = requests.get(url, timeout=30).json()
    if js.get("status") != "OK":
        raise RuntimeError(js)
    df = pd.DataFrame(js["results"])
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df.rename(columns={"t": "datetime", "o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", required=True)  # YYYY-MM-DD
    p.add_argument("--end", required=True)
    p.add_argument("--outdir", default="data")
    args = p.parse_args()
    Path(args.outdir).mkdir(exist_ok=True)
    df = fetch(args.symbol, args.start, args.end)
    outfile = Path(args.outdir) / f"{args.symbol}_{args.start}_{args.end}_1min.csv"
    df.to_csv(outfile, index=False)
    print(f"Saved {len(df):,} rows to {outfile}")

if __name__ == "__main__":
    main()