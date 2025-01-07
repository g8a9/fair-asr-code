#!/bin/bash

set -e

source /mnt/home/giuseppe/mydata/venvs/fairasr/bin/activate

if [ -z "$1" ]; then
    echo "Error: dataset argument is required"
    exit 1
fi
dataset=$1

# Main configs
input_dir="/mnt/home/giuseppe/myscratch/fair-asr-results/transcripts/"
output_dir="/mnt/home/giuseppe/myscratch/fair-asr-results/evaluation/"
group_contrast="F-M"

# Models
MODELS=( \
    "nvidia/canary-1b" \
    "openai/whisper-large-v3" \
    "openai/whisper-large-v3-turbo" \
    "facebook/seamless-m4t-v2-large" \
    "distil-whisper/distil-large-v3" \
)

for model in "${MODELS[@]}"; do
    
    python fair_asr_code/evaluate.py \
        --transcription_dir $input_dir \
        --model $model \
        --dataset $dataset \
        --output_dir $output_dir \
        --group_contrast $group_contrast

done
