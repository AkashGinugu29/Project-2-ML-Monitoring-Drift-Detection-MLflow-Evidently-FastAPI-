import os
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from config import DATA_REF_DIR, DATA_CUR_DIR, REPORTS_DIR

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    ref_path = os.path.join(DATA_REF_DIR, "reference.csv")
    cur_path = os.path.join(DATA_CUR_DIR, "current.csv")

    reference = pd.read_csv(ref_path)
    current = pd.read_csv(cur_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    out_path = os.path.join(REPORTS_DIR, "drift_report.html")
    report.save_html(out_path)

    print(f"✅ Drift report saved -> {out_path}")

if __name__ == "__main__":
    main()
