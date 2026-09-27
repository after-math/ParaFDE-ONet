"""Refit the manuscript's GBT from archived forward-training survivability labels.

This release utility uses the saved training-only tuning choices. It does not
rerun the historical hyperparameter search or train on application test labels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/grid_gbt_refit")
    args = parser.parse_args()
    source = ROOT / "results/paper/smart_grid"
    with np.load(source / "gbt_training_labels_from_forward_train.npz", allow_pickle=False) as archive:
        parameters = archive["parameters"]
        target = archive["survivability"]
        counts = archive["history_counts"]
        np.testing.assert_allclose(target, archive["safe_counts"] / counts, atol=1e-15)
        if parameters.shape != (1024, 3) or int(counts.sum()) != 24576:
            raise ValueError("Unexpected archived training design")
    settings = json.loads((source / "fair_gbt_five_tuning_seeds.json").read_text())
    application = pd.read_csv(ROOT / "results/paper/source_data/smart_grid_survivability_predictions.csv")
    # Only physical parameters are supplied to the trained regressor at prediction time.
    query = application[["K", "gamma", "alpha"]].to_numpy()
    rows = []
    predictions = []
    for run in settings["runs"]:
        model = GradientBoostingRegressor(random_state=int(run["cv_split_seed"]), **run["best_params"])
        model.fit(parameters, target)
        prediction = model.predict(query)
        # Test outcomes enter only the following evaluation, after fit and prediction.
        direct = application["direct_survivability"].to_numpy()
        difference = prediction - direct
        rows.append({
            "cv_split_seed": int(run["cv_split_seed"]),
            "mae_percentage_points": float(100 * np.mean(np.abs(difference))),
            "rmse_percentage_points": float(100 * np.sqrt(np.mean(difference**2))),
            "spearman": float(spearmanr(direct, prediction).statistic),
            "maximum_difference_from_archived_prediction": float(np.max(np.abs(prediction - np.asarray(run["prediction"])))),
        })
        predictions.append(prediction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
    result = application[["K", "gamma", "alpha"]].copy()
    for row, prediction in zip(rows, predictions):
        result[f"prediction_seed_{row['cv_split_seed']}"] = prediction
    result.to_csv(args.output_dir / "predictions.csv", index=False)
    print(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
