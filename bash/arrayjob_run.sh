#!/bin/bash
#SBATCH --job-name=fair_asr
#SBATCH --output=logs/%A-%a.out
#SBATCH --partition=a6000
#SBATCH --gpus=1
#SBATCH --qos=gpu-medium
#SBATCH --time=08:00:00
#SBATCH --array=0-112

config_id=${SLURM_ARRAY_TASK_ID}
output_dir="/mnt/home/giuseppe/myscratch/fair-asr-leaderboard/transcripts"

python src/running/transcribe_dataset.py \
    --dataset_id cv \
    --config_file cv_17_default.json \
    --config_id ${config_id} \
    --input_dir ~/myscratch/fair-asr-leaderboard/data/mozilla-foundation--common_voice_17_0/ \
    --output_dir ${output_dir} \
    --num_workers 4 --dry_run