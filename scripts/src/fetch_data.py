#!/usr/bin/env python3
"""Fetch 1‑minute OHLCV bars from Polygon.io and save to CSV."""

import argparse
import logging
import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class PolygonDataFetcher:
    """Enhanced Polygon.io data fetcher with comprehensive error handling."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLY_API_KEY")
        if not self.api_key:
            raise ValueError("POLY_API_KEY must be set in environment or .env file")

        self.base_url = (
            "https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/minute/{start}/{end}"
        )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "orderbook-alpha-signals/1.0"})

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 5 requests per second max

    def _validate_inputs(self, symbol: str, start_date: str, end_date: str) -> None:
        """Validate input parameters."""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")

        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError("Invalid date format. Use YYYY-MM-DD: %s", e)

        if start_dt >= end_dt:
            raise ValueError("Start date must be before end date")

        if end_dt > datetime.now():
            logger.warning("End date is in the future, adjusting to today")

        max_range = timedelta(days=730)  # 2 years max
        if end_dt - start_dt > max_range:
            raise ValueError("Date range too large. Maximum %d days", max_range.days)

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(self, url: str, max_retries: int = 3) -> Dict[Any, Any]:
        """Make HTTP request with retries and error handling."""
        for attempt in range(max_retries):
            try:
                self._rate_limit()

                response = self.session.get(url, timeout=30)
                response.raise_for_status()

                data = response.json()

                # Check API-specific error conditions
                if data.get("status") == "ERROR":
                    error_msg = data.get("error", "Unknown API error")
                    if "rate limit" in error_msg.lower():
                        logger.warning(
                            "Rate limit hit, retrying in %d seconds...", 2**attempt
                        )
                        time.sleep(2**attempt)
                        continue
                    raise requests.exceptions.RequestException(
                        "API Error: %s" % error_msg
                    )

                if data.get("status") != "OK":
                    logger.warning("API returned status: %s" % data.get("status"))
                    if data.get("results") is None:
                        raise requests.exceptions.RequestException(
                            "No data returned: %s" % data
                        )

                return data

            except requests.exceptions.Timeout:
                logger.warning(
                    "Request timeout (attempt %d/%d)" % (attempt + 1, max_retries)
                )
                if attempt == max_retries - 1:
                    raise
                time.sleep(2**attempt)

            except requests.exceptions.ConnectionError:
                logger.warning(
                    "Connection error (attempt %d/%d)" % (attempt + 1, max_retries)
                )
                if attempt == max_retries - 1:
                    raise
                time.sleep(2**attempt)

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = 2**attempt
                    logger.warning("Rate limited, waiting %d seconds..." % wait_time)
                    time.sleep(wait_time)
                    continue
                if e.response.status_code in [500, 502, 503, 504]:  # Server errors
                    logger.warning(
                        "Server error %d, retrying..." % e.response.status_code
                    )
                    time.sleep(2**attempt)
                    continue
                raise

        raise requests.exceptions.RequestException(
            "Failed after %d attempts" % max_retries
        )

    def fetch(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch OHLCV data with comprehensive error handling.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with OHLCV data

        Raises:
            ValueError: Invalid input parameters
            requests.exceptions.RequestException: Network/API errors
            pd.errors.EmptyDataError: No data returned
        """
        try:
            # Validate inputs
            self._validate_inputs(symbol, start_date, end_date)

            # Build URL
            url = self.base_url.format(
                sym=symbol.upper(), start=start_date, end=end_date
            )
            url += "?adjusted=true&limit=50000&sort=asc&apiKey=%s" % self.api_key

            logger.info(
                "Fetching %s data from %s to %s" % (symbol, start_date, end_date)
            )

            # Make request
            data = self._make_request(url)

            # Check for data
            if not data.get("results"):
                raise pd.errors.EmptyDataError(
                    "No data returned for %s between %s and %s"
                    % (symbol, start_date, end_date)
                )

            # Convert to DataFrame
            df = pd.DataFrame(data["results"])

            # Validate DataFrame structure
            required_columns = ["t", "o", "h", "l", "c", "v"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError("Missing required columns: %s" % missing_columns)

            # Process data
            df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
            df.rename(
                columns={
                    "t": "datetime",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                },
                inplace=True,
            )

            # Data quality checks
            if df.empty:
                raise pd.errors.EmptyDataError("Resulting DataFrame is empty")

            # Check for null values in critical columns
            critical_nulls = df[["open", "high", "low", "close"]].isnull().sum().sum()
            if critical_nulls > 0:
                logger.warning("Found %d null values in OHLC data" % critical_nulls)

            # Basic data validation
            invalid_prices = (df["high"] < df["low"]).sum()
            if invalid_prices > 0:
                logger.warning("Found %d rows where high < low" % invalid_prices)

            zero_volume = (df["volume"] == 0).sum()
            if zero_volume > 0:
                logger.info("Found %d rows with zero volume" % zero_volume)

            logger.info("Successfully fetched %d rows of data" % len(df))
            return df

        except Exception as e:
            logger.error("Error fetching data for %s: %s" % (symbol, e))
            raise


# Legacy function for backward compatibility
def fetch(sym: str, start: str, end: str) -> pd.DataFrame:
    """Legacy fetch function for backward compatibility."""
    fetcher = PolygonDataFetcher()
    return fetcher.fetch(sym, start, end)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", required=True)  # YYYY-MM-DD
    p.add_argument("--end", required=True)
    p.add_argument("--outdir", default="data")
    args = p.parse_args()
    Path(args.outdir).mkdir(exist_ok=True)
    df = fetch(args.symbol, args.start, args.end)
    outfile = (
        Path(args.outdir) / ("%s_%s_%s_1min.csv" % (args.symbol, args.start, args.end))
    )
    df.to_csv(outfile, index=False)
    print("Saved %d rows to %s" % (len(df), outfile))


if __name__ == "__main__":
    main()
