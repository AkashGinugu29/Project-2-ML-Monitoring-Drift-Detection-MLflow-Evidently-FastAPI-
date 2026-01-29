import os
import numpy as np
import pandas as pd

from config import DATA_REF_DIR, DATA_CUR_DIR

def make_reference(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    noise = rng.normal(0, 0.5, n)
    y = 3.0 * x1 + 0.8 * x2 + noise
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})

def make_current(n: int = 2000, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Drift: shift x2 distribution
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(7, 2.5, n)  # shifted mean + wider variance
    noise = rng.normal(0, 0.7, n)
    y = 3.0 * x1 + 0.8 * x2 + noise
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})

def main():
    os.makedirs(DATA_REF_DIR, exist_ok=True)
    os.makedirs(DATA_CUR_DIR, exist_ok=True)

    ref = make_reference()
    cur = make_current()

    ref.to_csv(os.path.join(DATA_REF_DIR, "reference.csv"), index=False)
    cur.to_csv(os.path.join(DATA_CUR_DIR, "current.csv"), index=False)

    print("✅ Generated reference and current datasets:")
    print(f"- data/reference/reference.csv ({len(ref)} rows)")
    print(f"- data/current/current.csv ({len(cur)} rows)")

if __name__ == "__main__":
    main()
