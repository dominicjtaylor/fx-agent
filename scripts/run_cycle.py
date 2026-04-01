"""
run_cycle.py
Entrypoint for a single fx-agent research cycle.

Loads OHLC data from a questdb-<pair>.csv file, then runs one full
propose -> test -> reason -> log cycle.

Usage:
    python scripts/run_cycle.py
    python scripts/run_cycle.py --hint "explore session timing effects"
    DATA_DIR=/path/to/data python scripts/run_cycle.py

Environment variables:
    DATA_DIR   Directory containing questdb-<pair>.csv files.
               Defaults to ../fx-volatility-forecasting/data relative to repo root.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Resolve repo root and add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.agent.context import load_context
from src.agent.loop import run_cycle


def load_pair_csv(data_dir: Path, pair: str) -> pd.DataFrame:
    """Load a questdb-<pair>.csv file for the given pair symbol."""
    filename = f"questdb-{pair.lower()}.csv"
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Expected questdb-{pair.lower()}.csv in {data_dir}\n"
            f"Set DATA_DIR env var to the directory containing your CSV files."
        )

    dtype = {
        "symbol": "category",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
    }
    df = pd.read_csv(path, parse_dates=["timestamp"], dtype=dtype)
    df = df.set_index("timestamp").sort_index()
    df = df[["open", "high", "low", "close"]]
    print(f"  Loaded {pair}: {len(df):,} rows  ({df.index[0]} → {df.index[-1]})")
    return df


def main():
    parser = argparse.ArgumentParser(description="Run one fx-agent research cycle")
    parser.add_argument(
        "--hint",
        type=str,
        default=None,
        help="Optional research hint passed to the proposal engine",
    )
    args = parser.parse_args()

    # Resolve data directory
    default_data_dir = REPO_ROOT.parent / "fx-volatility-forecasting" / "data"
    data_dir = Path(os.environ.get("DATA_DIR", str(default_data_dir)))

    print("fx-agent — Research Cycle")
    print("─" * 45)
    print(f"Data directory : {data_dir}")
    if args.hint:
        print(f"Research hint  : {args.hint}")
    print()

    # Load context to discover configured pairs
    context = load_context()
    pairs = context["data"]["pairs"]

    # Load data for each configured pair
    print("Loading data...")
    data = {}
    for pair in pairs:
        df = load_pair_csv(data_dir, pair)
        data[pair] = df
    print()

    # Run one full cycle
    run_cycle(data, user_hint=args.hint)


if __name__ == "__main__":
    main()
