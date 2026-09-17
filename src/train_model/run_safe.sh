# Setup conda environment and run training for RFFMG model
cd "$(cd "$(dirname "$0")" && pwd)/../.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate safe

# Settings for training
FRAG_NAME="rc_cms"          # "brics" or "rc_cms"
MODE="finetuning"          # "finetuning" or "from_scratch"
PRETRAINED_DIR="models/safe/gpt/pretrained"

if [ "$MODE" = "finetuning" ]; then
    MODEL_PATH_ARG="--model_path ${PRETRAINED_DIR}"
    OUTPUT_DIR="models/safe/gpt/finetuning/${FRAG_NAME}/"

elif [ "$MODE" = "from_scratch" ]; then
    MODEL_PATH_ARG=""
    OUTPUT_DIR="models/safe/gpt/from_scratch/${FRAG_NAME}/"

else
    echo "Unknown MODE: ${MODE} (use 'finetuning' or 'from_scratch')" >&2
    exit 1
fi

# wandb: separate local log directories by model, mode, and data slice for easy identification
export WANDB_MODE=offline
export WANDB_DIR="wandb/safe/gpt/${MODE}/${FRAG_NAME}"
mkdir -p "${WANDB_DIR}"

safe-train \
${MODEL_PATH_ARG} \
--config ${PRETRAINED_DIR}/config.json \
--tokenizer ${PRETRAINED_DIR}/tokenizer.json \
--dataset data/safe/${FRAG_NAME}/normal \
--output_dir ${OUTPUT_DIR} \
--text_column full_safe \
--num_train_epochs 50 \
--learning_rate 1e-4 \
--warmup_steps 10000 \
--do_train \
--do_eval \
--eval_strategy steps \
--per_device_train_batch_size 32 \
--eval_steps 5000 \
--save_strategy steps \
--save_steps 5000 \
--save_total_limit 5 \
--load_best_model_at_end
