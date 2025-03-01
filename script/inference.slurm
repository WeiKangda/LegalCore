#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=test   	#Set the job name to “JobExample4”
#SBATCH --time=6:00:00          	#Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1               	#Request 1 task
#SBATCH --mem=80G              	#Request 2560MB (2.5GB) per node
#SBATCH --output=test.%j  	#Send stdout/err to “Example4Out.[jobID]”
#SBATCH --gres=gpu:a100:2             	#Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu          	#Request the GPU partition/queue
##OPTIONAL JOB SPECIFICATIONS
##SBATCH --mail-type=ALL          	#Send email on all job events
##SBATCH --mail-user=email_address	#Send all emails to email_address

#First Executable Line
BASE_DIR="your_project"

conda activate legalcore
# Log in Huggingface using huggingface-cli, see https://huggingface.co/docs/huggingface_hub/en/guides/cli for details.

cd ${BASE_DIR}
PYTHON_FILE="${BASE_DIR}/baseline/run.py"
python ${PYTHON_FILE} --setting $1 --model_name $2 --inference_mode $3 --output_path $4 --data_path $5