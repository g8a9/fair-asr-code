#!/bin/bash
#SBATCH --job-name=fair_asr
#SBATCH --output=logs/%A-%a.out
#SBATCH --partition=a6000
#SBATCH --gpus=1
#SBATCH --qos=gpu-short
#SBATCH --time=04:00:00
#SBATCH --array=1-37

source /mnt/home/giuseppe/mydata/venvs/fairasr/bin/activate

# Pick a configuration file, run configs are stored there 
# config_file="config/cv_17_openai--whisper-large-v3_default.json"
# config_file="config/cv_17_openai--whisper-large-v3-turbo_default.json"
config_file="config/cv_17_distil-whisper--distil-large-v3_default.json"
# config_file="config/cv_17_facebook--seamless-m4t-v2-large_default.json"
# config_file="config/cv_17_nvidia--canary-1b_default.json"
# config_file="config/facebook--voxpopuli_nvidia--canary-1b_default.json"

config_id=${SLURM_ARRAY_TASK_ID}
output_dir="/mnt/home/giuseppe/myscratch/fair-asr-results/transcripts"
input_dir="/mnt/home/giuseppe/myscratch/fair-asr-leaderboard/data/mozilla-foundation--common_voice_17_0"

python fair_asr_code/transcribe_dataset.py \
    --dataset_id cv_17 \
    --config_file ${config_file} \
    --config_id ${config_id} \
    --output_dir ${output_dir} \
    --input_dir ${input_dir} \
    --num_workers 16