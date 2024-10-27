#!/bin/bash

# Parameters
NUM_JOBS=$1
#"/data/Dynamics/RealCaptureBlackBlueCloudRedBallSetData_cogvideox_dataset"
DATA_ROOT=$2

# Loop through device IDs
for device_id in {0..4}
do
  # Launch 3 jobs per device ID
  for job in {0..2}
  do
    # Calculate job_idx for each job
    job_idx=$((device_id * 3 + job))

    # Check if job_idx is within the range
    if [ "$job_idx" -lt "$NUM_JOBS" ]; then
      # Run the Python script with the appropriate arguments
      python forward_all_videos.py \
        --device_id "$device_id" \
        --job_idx "$job_idx" \
        --num_jobs "$NUM_JOBS" \
        --data_root "$DATA_ROOT" &
    fi
  done
done

# Wait for all background processes to complete
wait
echo "All jobs completed."
