from __future__ import annotations

import argparse
from pathlib import Path

from flight_prices.config import ARTIFACTS_DIR, DEFAULT_DATA_PATH
from flight_prices.train import train


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train flight fare regression models and write artifacts.")
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to Data_Train.xlsx / Airline.xlsx (Kaggle schema). Defaults to data/Data_Train.xlsx if present.",
    )
    p.add_argument("--out", type=Path, default=ARTIFACTS_DIR, help="Directory for model.joblib and metrics.json.")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--random-state", type=int, default=49)
    args = p.parse_args(argv)
    data_path = args.data
    if data_path is None:
        data_path = DEFAULT_DATA_PATH if DEFAULT_DATA_PATH.is_file() else None
    if data_path is None:
        p.error("Pass --data PATH.xlsx or place the Kaggle training file at data/Data_Train.xlsx")

    bundle = train(
        data_path,
        args.out,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Wrote artifacts to {args.out.resolve()}")
    print(f"Best model: {bundle['best_model']}")


if __name__ == "__main__":
    main()
