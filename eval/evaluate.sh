#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ekole-ml

# Usage: bash eval/evaluate.sh [cpu|gpu]
#   cpu  — run CPU-only pipeline (vio-metal)
#   gpu  — run Metal GPU pipeline (vio-metal-gpu) [default]

MODE="${1:-gpu}"

# Paths
DATASET_PATH="/Users/ekole/Datasets/vicon_room1/V1_01_easy"
GT_PATH="${DATASET_PATH}/mav0/state_groundtruth_estimate0/data.csv"
METALLIB_PATH="./build/shaders.metallib"

if [ "$MODE" = "cpu" ]; then
    EXECUTABLE="./build/vio-metal"
    echo "--- Running CPU Pipeline ---"
    $EXECUTABLE $DATASET_PATH --headless
else
    EXECUTABLE="./build/vio-metal-gpu"
    echo "--- Running GPU Pipeline ---"
    $EXECUTABLE $DATASET_PATH $METALLIB_PATH --headless
fi

# 2. Find the latest trajectory and cost log (by timestamp in filename)
EST_PATH=$(ls -t results/trajectories/estimated_*.txt 2>/dev/null | head -1)
COST_LOG=$(ls -t results/configs/cost_log_*.csv 2>/dev/null | head -1)

if [ -z "$EST_PATH" ]; then
    echo "Error: No estimated trajectory found in results/trajectories/"
    ls -R results
    exit 1
fi

echo "--- Using trajectory: $EST_PATH ---"
wc -l "$EST_PATH"

# 3. Convert EuRoC ground truth CSV to TUM format for evo
GT_TUM="results/configs/gt_tum.txt"
python3 -c "
import csv
with open('${GT_PATH}') as f:
    reader = csv.reader(f)
    header = next(reader)  # skip header
    with open('${GT_TUM}', 'w') as out:
        for row in reader:
            ts = float(row[0]) * 1e-9  # ns to seconds
            px, py, pz = row[1], row[2], row[3]
            qw, qx, qy, qz = row[4], row[5], row[6], row[7]
            # TUM format: timestamp tx ty tz qx qy qz qw
            out.write(f'{ts:.9f} {px} {py} {pz} {qx} {qy} {qz} {qw}\n')
"
echo "--- Ground truth converted to TUM: $GT_TUM ---"

# 4. Run EVO Evaluation
echo "--- Evaluating with EVO ---"

RUN_TAG=$(basename "$EST_PATH" | sed 's/estimated_//' | sed 's/.txt//')

# ATE (Absolute Trajectory Error)
evo_ape tum "$GT_TUM" "$EST_PATH" -va --plot --plot_mode xyz \
    --save_results "results/configs/ate_${RUN_TAG}.zip" \
    || echo "ATE evaluation failed"

# RPE (Relative Pose Error)
evo_rpe tum "$GT_TUM" "$EST_PATH" -v --plot --plot_mode xyz \
    --save_results "results/configs/rpe_${RUN_TAG}.zip" \
    || echo "RPE evaluation failed"

# 5. Plot cost function values
echo "--- Plotting Cost Function ---"
if [ -n "$COST_LOG" ]; then
    python3 eval/plot_cost.py "$COST_LOG"
else
    echo "No cost log found, skipping cost plot"
fi

echo "--- Evaluation Complete ---"
echo "Run tag: $RUN_TAG"
echo "Results in: results/configs/"
