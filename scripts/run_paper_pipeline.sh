#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash scripts/run_paper_pipeline.sh PDF [--collection NAME] [--tags a,b] [--publish]"
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

PDF_INPUT=$1
shift
COLLECTION="misc"
TAGS="paper"
PUBLISH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --collection)
      COLLECTION=${2:?"--collection requires a value"}
      shift 2
      ;;
    --tags)
      TAGS=${2:?"--tags requires a value"}
      shift 2
      ;;
    --publish)
      PUBLISH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SITE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
ENV_FILE="$SITE_ROOT/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Copy .env.example to .env and fill in the keys." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ -n "${BLABLAPAPER_DIR:-}" ]]; then
  BLABLA_ROOT=$BLABLAPAPER_DIR
elif [[ -f "$SITE_ROOT/BlaBlaPaper/main.py" ]]; then
  BLABLA_ROOT="$SITE_ROOT/BlaBlaPaper"
else
  BLABLA_ROOT="$SITE_ROOT/../BlaBlaPaper"
fi
if [[ ! -f "$BLABLA_ROOT/main.py" ]]; then
  echo "BlaBlaPaper not found at $BLABLA_ROOT" >&2
  echo "Set BLABLAPAPER_DIR in $ENV_FILE if it is elsewhere." >&2
  exit 1
fi

if [[ ! -f "$PDF_INPUT" ]]; then
  echo "PDF does not exist: $PDF_INPUT" >&2
  exit 1
fi
if [[ $(head -c 5 "$PDF_INPUT") != "%PDF-" ]]; then
  echo "Input does not look like a PDF: $PDF_INPUT" >&2
  exit 1
fi

for required_name in OPENAI_API_KEY MINERU_API_TOKEN; do
  required_value=${!required_name:-}
  if [[ -z "$required_value" || "$required_value" == REPLACE_WITH_* ]]; then
    echo "Fill $required_name in $ENV_FILE" >&2
    exit 1
  fi
done

IMAGE_MODEL_VALUE=${model_image:-${MODEL_NAME_IMAGE:-}}
if [[ -z "$IMAGE_MODEL_VALUE" || "$IMAGE_MODEL_VALUE" == REPLACE_WITH_* ]]; then
  echo "Fill model_image in $ENV_FILE with a vision-capable model." >&2
  exit 1
fi

SOURCE_SHA256=$(sha256sum "$PDF_INPUT" | cut -d ' ' -f 1)
WORK_ROOT="$SITE_ROOT/.pipeline-work/${SOURCE_SHA256:0:16}"
mkdir -p "$WORK_ROOT"
cp "$PDF_INPUT" "$WORK_ROOT/paper.pdf"

PIPELINE_VENV="$SITE_ROOT/.pipeline-venv"
if [[ ! -x "$PIPELINE_VENV/bin/python" ]]; then
  python3 -m venv "$PIPELINE_VENV"
  "$PIPELINE_VENV/bin/python" -m pip install --upgrade pip
  "$PIPELINE_VENV/bin/python" -m pip install -r "$BLABLA_ROOT/requirements.txt"
fi

(
  cd "$WORK_ROOT"
  "$PIPELINE_VENV/bin/python" "$BLABLA_ROOT/main.py" "$WORK_ROOT/paper.pdf"
)

REPORT_DIRS=()
if [[ -d "$WORK_ROOT/outputs" ]]; then
  while IFS= read -r candidate; do
    REPORT_DIRS+=("$candidate")
  done < <(find "$WORK_ROOT/outputs" -mindepth 1 -maxdepth 1 -type d -print)
fi
if [[ ${#REPORT_DIRS[@]} -ne 1 ]]; then
  echo "Expected exactly one generated report directory, found ${#REPORT_DIRS[@]}." >&2
  exit 1
fi

REPORT_DIR=${REPORT_DIRS[0]}
"$PIPELINE_VENV/bin/python" "$SCRIPT_DIR/import_blablapaper.py" \
  --source "$REPORT_DIR" \
  --content-root "$SITE_ROOT/content" \
  --collection "$COLLECTION" \
  --tags "$TAGS" \
  --source-sha256 "$SOURCE_SHA256"

(
  cd "$SITE_ROOT"
  if [[ ! -d node_modules ]]; then
    npm ci
  fi
  npx quartz plugin install --concurrency 2
  npx quartz build
)

SLUG=$(basename "$REPORT_DIR")
echo "Pipeline complete. Local site output: $SITE_ROOT/public"
echo "Paper route: $COLLECTION/$SLUG"

if [[ $PUBLISH -eq 1 ]]; then
  git -C "$SITE_ROOT" add content
  if git -C "$SITE_ROOT" diff --cached --quiet; then
    echo "No content changes to publish."
  else
    git -C "$SITE_ROOT" commit -m "paper: publish $SLUG"
    git -C "$SITE_ROOT" push origin HEAD
    echo "Pushed generated notes; the GitHub Pages deployment workflow will run automatically."
  fi
fi
