#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${PROJECT_DIR}/data/raw"
EXTRACT_DIR="${RAW_DIR}/myanimelist-2023-v3"
ARCHIVE_PATH="${RAW_DIR}/myanimelist-user-ratings-2023-v3.zip"
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/andrewgatchalian/myanimelist-user-ratings"
EXPECTED_SHA256="1a86c041de4505dc651c7c6c3bde6d07ba9a695cf3367725a095cb8eef57fbfd"

mkdir -p "${RAW_DIR}" "${EXTRACT_DIR}"

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  curl -L --fail --retry 3 -C - -o "${ARCHIVE_PATH}" "${DATASET_URL}"
fi

ACTUAL_SHA256="$(openssl dgst -sha256 "${ARCHIVE_PATH}" | awk '{print $NF}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
  echo "Dataset checksum mismatch: expected ${EXPECTED_SHA256}, got ${ACTUAL_SHA256}" >&2
  exit 1
fi

unzip -n -q "${ARCHIVE_PATH}" \
  anime_data/anime_titles.csv \
  anime_data/anime_user_ratings.csv \
  -d "${EXTRACT_DIR}"

for required_file in anime_titles.csv anime_user_ratings.csv; do
  if [[ ! -f "${EXTRACT_DIR}/anime_data/${required_file}" ]]; then
    echo "Missing expected dataset file: ${required_file}" >&2
    exit 1
  fi
done

echo "Dataset ready at ${EXTRACT_DIR}/anime_data"
