#!/bin/bash
#
# Set the base directory and output paths
BASE_DIR="your_project" # Change the BASE_DIR to your project path

CURRENT_MONTH=$(date +%m)
CURRENT_DAY=$(date +%d)
OUTPUT_PATH="${BASE_DIR}/output/${CURRENT_MONTH}/${CURRENT_DAY}/"

mkdir -p ${OUTPUT_PATH}
echo "Output directory: ${OUTPUT_PATH}"
cd ${OUTPUT_PATH}
# Declare task-specific parameters
# Change based on your needs
declare -a tasks=("event_detection" "event_coreference" "end2end")
declare -a models=("Llama-3.1-8b-instruct" "Mistral-7b" "QWen-7b" "QWen-14b" "Phi" "Phi-small" "GPT-4-Turbo" "Mistral-Nemo")
declare -a prompts=("zero_shot" "one_shot" "two_shot")

DATA_PATH="./data/data.jsonl"

COUNT=0

# Loop through tasks, models, and prompts
for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        for prompt in "${prompts[@]}"; do
            OUTPUT_FILE="${OUTPUT_PATH}${task}_${model}_${prompt}"
            ERROR_FILE="${OUTPUT_PATH}${task}_${model}_${prompt}"
            SLURM_FILE="${BASE_DIR}/script/inference.slurm"
            # Submit the job using sbatch or directly execute
            sbatch --output=${OUTPUT_FILE}.%j -J "${task}_${model}_${prompt}" ${SLURM_FILE} $task $model $prompt $OUTPUT_PATH $DATA_PATH
            echo "Submitted: ${task} | Model: ${model} | Prompt: ${prompt}"
            COUNT=$((COUNT + 1))
        done
    done
done

echo "$COUNT jobs submitted."