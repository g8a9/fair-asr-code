#!/bin/bash
#SBATCH --job-name=fair_asr
#SBATCH --output=logs/%A-%a.out
#SBATCH --partition=a6000
#SBATCH --gpus=1
#SBATCH --qos=gpu-short
#SBATCH --time=04:00:00
#SBATCH --array=0-21

source /mnt/home/giuseppe/mydata/venvs/fairasr/bin/activate

# Pick a configuration file, run configs are stored there 
config_file="config/facebook--voxpopuli_nvidia--canary-1b_default.json"
# config_file="config/facebook--voxpopuli_facebook--seamless-m4t-v2-large_default.json"
# config_file="config/facebook--voxpopuli_openai--whisper-large-v3_default.json"
# config_file="config/facebook--voxpopuli_openai--whisper-large-v3-turbo_default.json"

config_id=${SLURM_ARRAY_TASK_ID}
output_dir="/mnt/home/giuseppe/myscratch/fair-asr-results/transcripts"

python running/transcribe_dataset.py \
    --dataset_id facebook/voxpopuli \
    --config_file ${config_file} \
    --config_id ${config_id} \
    --output_dir ${output_dir} \
    --num_workers 16 \
    --speaker_id_col speaker_id \
    --reference_col raw_text