#!/bin/zsh

# Script to run calibration pipeline for a specific camera
# Usage: ./run_calib.sh <camera_number>
# Example: ./run_calib.sh 02

set -e

# Get the directory where this script is located
SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

# Check for camera number argument
if [[ -z "$1" ]]; then
    echo "Usage: $0 <camera_number>"
    echo "Available cameras: 02, 03, 04, 05, 06, 07, 08"
    exit 1
fi

CAM_NUM="$1"

# Map camera number to raw JSON filename
get_raw_json_file() {
    case "$1" in
        02) echo "fwc_c.json" ;;      # FWC_C - Front Wide Camera Center
        03) echo "fnc_new.json" ;;    # FNC - Front Narrow Camera
        04) echo "rnc_r.json" ;;      # RNC_R - Rear Narrow Camera Right
        05) echo "FWC_R.json" ;;      # FWC_R - Front Wide Camera Right
        06) echo "rnc_c.json" ;;      # RNC_C - Rear Narrow Camera Center
        07) echo "FWC_L.json" ;;      # FWC_L - Front Wide Camera Left
        08) echo "rnc_l.json" ;;      # RNC_L - Rear Narrow Camera Left
        *)  echo "" ;;
    esac
}

RAW_JSON_FILE=$(get_raw_json_file "$CAM_NUM")

# Validate camera number
if [[ -z "$RAW_JSON_FILE" ]]; then
    echo "Error: Invalid camera number '$CAM_NUM'"
    echo "Available cameras: 02, 03, 04, 05, 06, 07, 08"
    exit 1
fi

RAW_JSON="data/raw_jsons/$RAW_JSON_FILE"

# Check if raw JSON file exists
if [[ ! -f "$RAW_JSON" ]]; then
    echo "Error: Raw JSON file not found: $RAW_JSON"
    exit 1
fi

# Read offsets from test_params.json
PARAMS_FILE="test_params.json"
if [[ ! -f "$PARAMS_FILE" ]]; then
    echo "Error: Parameters file not found: $PARAMS_FILE"
    exit 1
fi

ROLL=$(python3 -c "import json; print(json.load(open('$PARAMS_FILE')).get('roll', 0.0))")
PITCH=$(python3 -c "import json; print(json.load(open('$PARAMS_FILE')).get('pitch', 0.0))")
YAW=$(python3 -c "import json; print(json.load(open('$PARAMS_FILE')).get('yaw', 0.0))")

echo "=== Calibration Pipeline for cam$CAM_NUM ==="
echo "Input: $RAW_JSON"
echo "Offsets: roll=$ROLL, pitch=$PITCH, yaw=$YAW"
echo "Params file: $PARAMS_FILE"
echo ""

# Step 1: Activate virtual environment if it exists
VENV_ACTIVATED=false
for VENV_DIR in "venv" ".venv" "env"; do
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        echo "Activating virtual environment: $VENV_DIR"
        source "$VENV_DIR/bin/activate"
        VENV_ACTIVATED=true
        break
    fi
done

# Also check VENV_PATH environment variable
if [[ "$VENV_ACTIVATED" = false && -n "$VENV_PATH" && -f "$VENV_PATH/bin/activate" ]]; then
    echo "Activating virtual environment from VENV_PATH: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    VENV_ACTIVATED=true
fi

if [[ "$VENV_ACTIVATED" = false ]]; then
    echo "Warning: No virtual environment found. Using system Python."
fi

# Step 2: Run calibration conversion
OUTPUT_CALIB="data/cam${CAM_NUM}/calib.json"

echo ""
echo "=== Step 1: Converting calibration ==="
echo "Running: python scripts/convert_lucid_calib.py --input $RAW_JSON --output $OUTPUT_CALIB --roll-offset $ROLL --pitch-offset $PITCH --yaw-offset $YAW --params-file $PARAMS_FILE"
echo ""

python scripts/convert_lucid_calib.py \
    --input "$RAW_JSON" \
    --output "$OUTPUT_CALIB" \
    --roll-offset "$ROLL" \
    --pitch-offset "$PITCH" \
    --yaw-offset "$YAW" \
    --params-file "$PARAMS_FILE"

# Step 3: Run lidar2camera
echo ""
echo "=== Step 2: Running lidar2camera ==="
echo "Running: ./bin/run_lidar2camera $OUTPUT_CALIB"
echo ""

./bin/run_lidar2camera "$OUTPUT_CALIB"

echo ""
echo "=== Calibration pipeline complete ==="
