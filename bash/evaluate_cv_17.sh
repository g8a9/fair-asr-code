#!/bin/bash



dataset="cv_17"
input_dir="/mnt/home/giuseppe/myscratch/fair-asr-results/transcripts/${dataset}"
output_dir="/mnt/home/giuseppe/myscratch/fair-asr-results/evaluation/${dataset}"

target_col="gender"
minority_group="female_feminine"
majority_group="male_masculine"

MODELS=( "openai/whisper-large-v3" "openai/whisper-large-v3-turbo" )
LANGS=( "de" "en" "nl" "ru" "sr" "it" "fr" "es" "ca" "pt" "sw" "yo" "ja" "hu" "fi" "ro" "cs" "sk" "ar" )

for lang in "${LANGS[@]}"; do
    for model in "${MODELS[@]}"; do
        
        python running/evaluate.py \
            --lang $lang \
            --transcription_dir $input_dir \
            --model $model \
            --dataset $dataset \
            --output_dir $output_dir \
            --target_col $target_col \
            --minority_group $minority_group \
            --majority_group $majority_group

    done
done
