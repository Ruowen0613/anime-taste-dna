#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${PROJECT_DIR}/data/raw"
EXTRACT_DIR="${RAW_DIR}/cooperunion-anime-v1"
ARCHIVE_PATH="${RAW_DIR}/anime-recommendations-database-v1.zip"
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/CooperUnion/anime-recommendations-database"
EXPECTED_SHA256="49595ab309d1d773ae66464379dc23d813466199816ae6b24e38bc73b8f2cd12"

mkdir -p "${RAW_DIR}" "${EXTRACT_DIR}"

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  curl -L --fail --retry 3 -o "${ARCHIVE_PATH}" "${DATASET_URL}"
fi

ACTUAL_SHA256="$(openssl dgst -sha256 "${ARCHIVE_PATH}" | awk '{print $NF}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
  echo "Dataset checksum mismatch: expected ${EXPECTED_SHA256}, got ${ACTUAL_SHA256}" >&2
  exit 1
fi

unzip -n -q "${ARCHIVE_PATH}" -d "${EXTRACT_DIR}"

for required_file in anime.csv rating.csv; do
  if [[ ! -f "${EXTRACT_DIR}/${required_file}" ]]; then
    echo "Missing expected dataset file: ${required_file}" >&2
    exit 1
  fi
done

echo "Dataset ready at ${EXTRACT_DIR}"
