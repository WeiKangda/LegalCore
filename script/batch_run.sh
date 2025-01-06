#!/bin/bash
#
# Set the base directory and output paths
#BASE_DIR="/scratch/user/your_project"
BASE_DIR="/scratch/user/xishi/llm/event_extract"

CURRENT_MONTH=$(date +%m)
CURRENT_DAY=$(date +%d)
OUTPUT_PATH="${BASE_DIR}/output/${CURRENT_MONTH}/${CURRENT_DAY}/"
mkdir -p ${OUTPUT_PATH}
echo "Output directory: ${OUTPUT_PATH}"
cd ${OUTPUT_PATH}
# Declare task-specific parameters
# declare -a tasks=("event_detection" "event_coreference" "end2end")
declare -a tasks=("event_coreference")

# declare -a models=("Llama-3.1-8b-instruct" "Mistral-7b" "QWen" "Phi" "GPT-4-Turbo")
# declare -a models=("QWen" "Phi" "Phi-small" "GPT-4-Turbo" "Mistral-7b")
declare -a models=("Llama-3.1-8b-instruct" "GPT-4-Turbo")

# declare -a prompts=("zero_shot" "one_shot" "two_shot")
declare -a prompts=("zero_shot")
# Declare additional parameters
DATA_PATH="./annotation_validation/jonathan_annotations/data.jsonl"
COUNT=0

# Loop through tasks, models, and prompts
for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        for prompt in "${prompts[@]}"; do
            OUTPUT_FILE="${OUTPUT_PATH}${task}_${model}_${prompt}"
            ERROR_FILE="${OUTPUT_PATH}${task}_${model}_${prompt}"
            SLURM_FILE="${BASE_DIR}/script/inference.slurm"
            # Submit the job using sbatch or directly execute
            sbatch --output=${OUTPUT_FILE}.%j -J "${task}_${model}_${prompt}" ${SLURM_FILE} $task $model $prompt $OUTPUT_PATH
            echo "Submitted: ${task} | Model: ${model} | Prompt: ${prompt}"
            COUNT=$((COUNT + 1))
        done
    done
done

echo "$COUNT jobs submitted."


##!/bin/bash
##
#cd /scratch/user/xishi/llm/vlm_attack
#current_month=$(date +%m)
#current_day=$(date +%d)
#output_path=output/${current_month}/${current_day}/
#mkdir -p ${output_path}
#echo "output directory: ${output_path}"
#
#cd ${output_path}
#output_path=../${output_path}
#echo "output directory: ${output_path}"
#
#
#
#
#w2s_mode="v4.1"
## # declare -a seeds=("42")
#
#count=0
#
#
#
#
## python testest.py --output_path ../${output_path}
#
#
##base generate (will only use tgt_model to generate)
##declare -a eval_modes=("w2s","sft","base","adv_w2s","adv_base","def_base","def_w2s") #
## declare -a eval_modes=("base" "adv_base" "def_base") #
#
## declare -a N=("0")
## declare -a betas=("0")
## declare -a tgt_models=("qwen0.5b" "qwen7b") #("qwen0.5b") #("llama3") "qwen7b" ("qwen72b" "qwen110b")
## declare -a base_models=("qwen7b") #=["vicuna_13b","vicuna_7b"
#
#
#
#
## for eval_mode in "${eval_modes[@]}"
## do
##    for N in "${N[@]}"
##    do
##        for beta in "${betas[@]}"
##        do
##             for tgt_model in "${tgt_models[@]}"
##             do
##                 for base_model in "${base_models[@]}"
##                 do
##                     sbatch --output=generate_${tgt_model}_${w2s_mode}_${eval_mode}_${beta}_N${N}.%j -J generate_$tgt_model_$w2s_mode_$eval_mode /scratch/user/xishi/llm/vlm_attack/utils/generate_v3.sh $beta $N $eval_mode $w2s_mode $tgt_model $base_model $output_path
##                     count=$((count+1))
##                 done
##             done
##        done
##    done
## done
#
#
## w2s generate
#declare -a eval_modes=("w2s","sft","base","adv_w2s","adv_base","def_base","def_w2s") #
#declare -a eval_modes=("adv_w2s") #
#
#declare -a N=("1024")
#declare -a betas=("1.5")
#declare -a tgt_models=("qwen7b") #("vicuna_7b" "vicuna_13b") #("qwen7b") #("34b" "mistral-7b")
#declare -a base_models=("qwen0.5b") #=["vicuna_13b","vicuna_7b"
#
#
#
#
#
## for eval_mode in "${eval_modes[@]}"
## do
##    for N in "${N[@]}"
##    do
##        for beta in "${betas[@]}"
##        do
##             for tgt_model in "${tgt_models[@]}"
##             do
##                 for base_model in "${base_models[@]}"
##                 do
##                     sbatch --output=generate_${tgt_model}_${w2s_mode}_${eval_mode}_${beta}_N${N}.%j -J generate_$tgt_model_$w2s_mode_$eval_mode /scratch/user/xishi/llm/vlm_attack/utils/generate_v3.sh $beta $N $eval_mode $w2s_mode $tgt_model $base_model $output_path
##                     count=$((count+1))
##                 done
##             done
##        done
##    done
## done
#
#declare -a eval_modes=("def_w2s") #
#
#declare -a N=("10")
## declare -a betas=("1" "3" "5")
#declare -a betas=("1")
#declare -a tgt_models=("qwen7b") #("vicuna_13b") #("qwen7b") #("34b" "mistral-7b")
#declare -a base_models=("qwen7b") #("vicuna_13b") #=["vicuna_13b","vicuna_7b"
#
#
#
#
#
#for eval_mode in "${eval_modes[@]}"
#do
#   for N in "${N[@]}"
#   do
#       for beta in "${betas[@]}"
#       do
#            for tgt_model in "${tgt_models[@]}"
#            do
#                for base_model in "${base_models[@]}"
#                do
#                    sbatch --output=generate_${tgt_model}_${w2s_mode}_${eval_mode}_${beta}_N${N}.%j -J generate_$tgt_model_$w2s_mode_$eval_mode /scratch/user/xishi/llm/vlm_attack/utils/generate_v3.sh $beta $N $eval_mode $w2s_mode $tgt_model $base_model $output_path
#                    count=$((count+1))
#                done
#            done
#       done
#   done
#done
#
#
#declare -a eval_modes=("def_w2s") #
#
#declare -a N=("10")
#declare -a betas=()
#for i in $(seq 0.1 0.1 1); do
#    betas+=("$i")
#done
#
## declare -a betas=("1")
#declare -a tgt_models=("vicuna_7b") #("vicuna_13b") #("qwen7b") #("34b" "mistral-7b")
#declare -a base_models=("vicuna_7b") #("vicuna_13b") #=["vicuna_13b","vicuna_7b"
#
#
#
#
#
#for eval_mode in "${eval_modes[@]}"
#do
#   for N in "${N[@]}"
#   do
#       for beta in "${betas[@]}"
#       do
#            for tgt_model in "${tgt_models[@]}"
#            do
#                for base_model in "${base_models[@]}"
#                do
#                    sbatch --output=generate_${tgt_model}_${w2s_mode}_${eval_mode}_${beta}_N${N}.%j -J generate_$tgt_model_$w2s_mode_$eval_mode /scratch/user/xishi/llm/vlm_attack/utils/generate_v3.sh $beta $N $eval_mode $w2s_mode $tgt_model $base_model $output_path
#                    count=$((count+1))
#                done
#            done
#       done
#   done
#done
#echo "$count jobs submitted."
