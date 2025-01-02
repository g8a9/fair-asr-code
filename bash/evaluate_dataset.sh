#!/bin/bash

set -e

source /mnt/home/giuseppe/mydata/venvs/fairasr/bin/activate

if [ -z "$1" ]; then
    echo "Error: dataset argument is required"
    exit 1
fi
dataset=$1
input_dir="/mnt/home/giuseppe/myscratch/fair-asr-results/transcripts/"
output_dir="/mnt/home/giuseppe/myscratch/fair-asr-results/evaluation/"
# Main configs
group_contrast="F-M"

# Models
MODELS=( \
    "nvidia/canary-1b" \
    "openai/whisper-large-v3" \
    "openai/whisper-large-v3-turbo" \
    "facebook/seamless-m4t-v2-large" \
)

for model in "${MODELS[@]}"; do
    
    python src/evaluate.py \
        --transcription_dir $input_dir \
        --model $model \
        --dataset $dataset \
        --output_dir $output_dir \
        --group_contrast $group_contrast

done
