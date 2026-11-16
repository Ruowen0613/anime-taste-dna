from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def export_signed_hybrid(
    source_artifact: Path,
    signed_test_report: Path,
    rating_scale_test_report: Path,
    output_artifact: Path,
) -> None:
    signed_report = json.loads(signed_test_report.read_text(encoding="utf-8"))
    if not signed_report.get("accepted_for_production"):
        raise ValueError("Signed hybrid did not pass the frozen production gates")
    rating_scale_report = json.loads(rating_scale_test_report.read_text(encoding="utf-8"))
    if not rating_scale_report.get("accepted_for_production"):
        raise ValueError("Rating-scale popularity decay did not pass the frozen production gates")
    signed_selection = signed_report["fixed_selection"]
    rating_scale_selection = rating_scale_report["frozen_selection"]
    with np.load(source_artifact, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays.update(
        {
            "source_model_version": arrays["model_version"],
            "model_version": np.asarray(["signed-rating-hybrid-v2"], dtype=np.str_),
            "signed_fold_in_regularization": np.asarray(
                [signed_selection["signed_regularization"]], dtype=np.float32
            ),
            "signed_score_weight": np.asarray(
                [signed_selection["signed_weight"]], dtype=np.float32
            ),
            "popularity_decay_pivot": np.asarray(
                [rating_scale_selection["popularity_decay_pivot"]], dtype=np.int32
            ),
            "popularity_decay_exponent": np.asarray(
                [rating_scale_selection["popularity_decay_exponent"]], dtype=np.float32
            ),
        }
    )
    output_artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_artifact, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the accepted signed-rating hybrid")
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--signed-test-report", type=Path, required=True)
    parser.add_argument("--rating-scale-test-report", type=Path, required=True)
    parser.add_argument("--output-artifact", type=Path, required=True)
    args = parser.parse_args()
    export_signed_hybrid(
        args.source_artifact.resolve(),
        args.signed_test_report.resolve(),
        args.rating_scale_test_report.resolve(),
        args.output_artifact.resolve(),
    )


if __name__ == "__main__":
    main()
