# Setup conda environment and run training for PromptSMILES model
cd "$(cd "$(dirname "$0")" && pwd)/../.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate promptsmiles

# Settings for training
FRAG_NAME="brics"          # "brics" or "rc_cms"
MODE="finetuning"          # "finetuning" or "from_scratch"

# wandb: separate local log directories by model, mode, and data slice for easy identification
export WANDB_MODE=offline
export WANDB_DIR="wandb/promptsmiles/gpt/${MODE}/${FRAG_NAME}"
mkdir -p "${WANDB_DIR}"

python src/train_model/train_promptsmiles.py \
    --frag_method "${FRAG_NAME}" \
    --mode "${MODE}" \
    --num_train_epochs 50 \
    --learning_rate 1e-4 \
    --warmup_steps 10000 \
    --eval_strategy steps \
    --eval_steps 5000 \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 5 \
    --max_length 256 \
    --per_device_train_batch_size 32
