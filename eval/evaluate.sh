#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ekole

# Paths
DATASET_PATH="/Users/ekole/Datasets/vicon_room1/V1_01_easy"
GT_PATH="${DATASET_PATH}/mav0/state_groundtruth_estimate0/data.csv"
EST_PATH="results/trajectories/estimated.txt"
METALLIB_PATH="./build/shaders.metallib" 
EXECUTABLE="./build/vio-metal-gpu"       

# Create results directory if it doesn't exist
mkdir -p results/trajectories

# 2. Run the VIO Pipeline
echo "--- Running VIO Pipeline ---"
$EXECUTABLE $DATASET_PATH $METALLIB_PATH

# 3. Check if estimation file was created
if [ ! -f "$EST_PATH" ]; then
    echo "Error: Estimation file not found at $EST_PATH"
    # List files to see where it went
    ls -R results
    exit 1
fi

# 4. Run EVO Evaluation
echo "--- Evaluating with EVO ---"

# IMPORTANT: 
# - We use 'euroc' for the GT_PATH because it's the raw data.csv
# - We use 'tum' for the EST_PATH because your TrajectoryWriter uses spaces.
# - --align_origin is good, but --align (Sim3) is better if you have scale drift.

# ATE (Absolute Trajectory Error)

evo_ape euroc $GT_PATH $EST_PATH -v --align --plot --plot_mode xyz --save_results results/ate_results.zip

# RPE (Relative Pose Error)

evo_rpe euroc $GT_PATH $EST_PATH -v --plot --plot_mode xyz --save_results results/rpe_results.zip

echo "--- Evaluation Complete ---"