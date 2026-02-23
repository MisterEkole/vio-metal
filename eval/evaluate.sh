#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ekole
# 1. Setup Paths
DATASET_PATH="/Users/ekole/Datasets/vicon_room1/V1_01_easy"
GT_PATH="${DATASET_PATH}/mav0/state_groundtruth_estimate0/data.csv"
EST_PATH="results/trajectories/estimated.txt"
METALLIB_PATH="./build/undistort.metallib"
EXECUTABLE="./build/vio-metal"

# 2. Run the VIO Pipeline
echo "--- Running VIO Pipeline ---"
$EXECUTABLE $DATASET_PATH $METALLIB_PATH

# 3. Check if estimation file was created
if [ ! -f "$EST_PATH" ]; then
    echo "Error: Estimation file not found at $EST_PATH"
    exit 1
fi

# 4. Run EVO Evaluation
echo "--- Evaluating with EVO ---"

# ATE (Absolute Trajectory Error) - Overall accuracy
evo_ape euroc $GT_PATH $EST_PATH -v --align_origin --plot --plot_mode xy --save_results results/ate_results.zip

# RPE (Relative Pose Error) - Local consistency/drift per meter
evo_rpe euroc $GT_PATH $EST_PATH -v --plot --plot_mode xyz --save_results results/rpe_results.zip

echo "--- Evaluation Complete ---"