#!/usr/bin/env bash
set -euo pipefail

# Download Bridge2AI Voice from PhysioNet with optional static-only filtering.
#
# Examples:
#   PHYSIONET_USER="your_user" scripts/download_physionet_b2ai.sh
#   PHYSIONET_USER="your_user" scripts/download_physionet_b2ai.sh --mode all
#   scripts/download_physionet_b2ai.sh --user your_user --dest raw_data --base-url https://physionet.org/files/b2ai-voice/3.0.0/

BASE_URL="https://physionet.org/files/b2ai-voice/3.0.0/"
DEST_DIR="raw_data"
MODE="static"  # static|all
USER_NAME="${PHYSIONET_USER:-}"
PASSWORD="${PHYSIONET_PASSWORD:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      USER_NAME="${2:-}"
      shift 2
      ;;
    --password)
      PASSWORD="${2:-}"
      shift 2
      ;;
    --dest)
      DEST_DIR="${2:-}"
      shift 2
      ;;
    --base-url)
      BASE_URL="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: download_physionet_b2ai.sh [options]

Options:
  --user <username>    PhysioNet username (or set PHYSIONET_USER env var)
  --dest <path>        Destination root directory (default: raw_data)
  --base-url <url>     Dataset base URL (default: https://physionet.org/files/b2ai-voice/3.0.0/)
  --mode <static|all>  static: only phenotype/static TSV files, all: full recursive download
  -h, --help           Show this help
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$USER_NAME" ]]; then
  echo "Missing PhysioNet username. Use --user or set PHYSIONET_USER." >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

WGET_COMMON=(
  -r -N -c -np
  -P "$DEST_DIR"
)

# Prefer direct auth flags for non-interactive runs to avoid fragile
# process-substitution config files on some shells/platforms.
if [[ -n "$PASSWORD" ]]; then
  WGET_AUTH=(--user "$USER_NAME" --password "$PASSWORD")
else
  WGET_AUTH=(--user "$USER_NAME" --ask-password)
fi

if [[ "$MODE" == "all" ]]; then
  echo "Downloading full dataset tree from: $BASE_URL"
  wget "${WGET_AUTH[@]}" "${WGET_COMMON[@]}" "$BASE_URL"
  exit 0
fi

if [[ "$MODE" != "static" ]]; then
  echo "Invalid --mode value: $MODE (use 'static' or 'all')." >&2
  exit 1
fi

echo "Downloading static-only files from: $BASE_URL"
wget \
  "${WGET_AUTH[@]}" \
  "${WGET_COMMON[@]}" \
  --accept-regex '.*(phenotype/|features/|phenotype[^/]*\.tsv|static[^/]*features[^/]*\.tsv)$' \
  --reject-regex '.*(mfcc|spectrogram).*' \
  "$BASE_URL"
