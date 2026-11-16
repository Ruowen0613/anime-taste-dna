from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def export_signed_hybrid(
    source_artifact: Path,
    test_report: Path,
    output_artifact: Path,
) -> None:
    report = json.loads(test_report.read_text(encoding="utf-8"))
    if not report.get("accepted_for_production"):
        raise ValueError("Signed hybrid did not pass the frozen production gates")
    selection = report["fixed_selection"]
    with np.load(source_artifact, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays.update(
        {
            "source_model_version": arrays["model_version"],
            "model_version": np.asarray(["signed-rating-hybrid-v1"], dtype=np.str_),
            "signed_fold_in_regularization": np.asarray(
                [selection["signed_regularization"]], dtype=np.float32
            ),
            "signed_score_weight": np.asarray(
                [selection["signed_weight"]], dtype=np.float32
            ),
        }
    )
    output_artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_artifact, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the accepted signed-rating hybrid")
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    parser.add_argument("--output-artifact", type=Path, required=True)
    args = parser.parse_args()
    export_signed_hybrid(
        args.source_artifact.resolve(),
        args.test_report.resolve(),
        args.output_artifact.resolve(),
    )


if __name__ == "__main__":
    main()
