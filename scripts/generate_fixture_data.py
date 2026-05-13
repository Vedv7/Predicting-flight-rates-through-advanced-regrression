"""Generate a small synthetic dataset that matches the Kaggle-style schema (for CI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AIRLINES = ["IndiGo", "Air India", "SpiceJet", "Vistara", "Go Air", "Jet Airways"]
SOURCES = ["Bangalore", "New Delhi", "Kolkata", "Chennai", "Mumbai"]
DESTINATIONS = ["New Delhi", "Bangalore", "Cochin", "Hyderabad", "Kolkata"]
STOPS = ["non-stop", "1 stop", "2 stops", "2+-stop"]

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 220
    rows = []
    for _ in range(n):
        airline = AIRLINES[int(rng.integers(0, len(AIRLINES)))]
        src = SOURCES[int(rng.integers(0, len(SOURCES)))]
        dst = DESTINATIONS[int(rng.integers(0, len(DESTINATIONS)))]
        if dst == src:
            dst = DESTINATIONS[(int(rng.integers(0, len(DESTINATIONS))) + 1) % len(DESTINATIONS)]
        stops = STOPS[int(rng.integers(0, len(STOPS)))]
        h = int(rng.integers(1, 10))
        m = int(rng.choice([0, 15, 25, 30, 45]))
        duration = f"{h}h {m}m"
        day = int(rng.integers(1, 28))
        month_num = int(rng.integers(3, 7))
        year = 2019
        doj = f"{day:02d}/{month_num:02d}/{year}"
        dep_h = int(rng.integers(0, 23))
        dep_m = int(rng.choice([0, 15, 30, 45]))
        arr_h = (dep_h + h) % 24
        arr_m = (dep_m + m) % 60
        month_abbr = ["Mar", "Apr", "May", "Jun"][month_num - 3]
        arrival = f"{arr_h:02d}:{arr_m:02d} {day:02d} {month_abbr}"
        dep_time = f"{dep_h:02d}:{dep_m:02d}"
        base = 2500 + h * 420 + (0 if stops == "non-stop" else 900)
        noise = float(rng.normal(0, 400))
        price = max(1800, int(base + noise))
        rows.append(
            {
                "Airline": airline,
                "Date_of_Journey": doj,
                "Source": src,
                "Destination": dst,
                "Route": f"{src[:3]} → {dst[:3]}",
                "Dep_Time": dep_time,
                "Arrival_Time": arrival,
                "Duration": duration,
                "Total_Stops": stops,
                "Additional_Info": "No info",
                "Price": price,
            }
        )
    out = pd.DataFrame(rows)
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "airline_synthetic.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Wrote {path} ({len(out)} rows)")
