from __future__ import annotations

import argparse
from pathlib import Path

from app.model_artifact_verification import verify_model_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify packaged recommendation artifact")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    verify_model_artifact(args.artifact.resolve(), args.manifest.resolve())
    print(f"Verified {args.artifact} against {args.manifest}")


if __name__ == "__main__":
    main()
