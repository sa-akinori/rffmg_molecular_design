# Setup conda environment and run training for RFFMG model
cd "$(cd "$(dirname "$0")" && pwd)/../.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rffmg

# Settings for training
FRAG_NAME="brics"          # "brics" or "rc_cms"
MODE="finetuning"          # "finetuning" or "from_scratch"
MODEL_NAME="t5chem"        # "t5chem" or "gpt"
SAMPLING_NUM=10            # 5 or 10 (uses the data/rffmg/<frag>/<N>times_sampling slice)

if [ "$MODE" != "finetuning" ] && [ "$MODE" != "from_scratch" ]; then
    echo "Unknown MODE: ${MODE} (use 'finetuning' or 'from_scratch')" >&2
    exit 1
fi

SAMPLING="${SAMPLING_NUM}times_sampling"
OUTPUT_DIR="models/rffmg/${MODEL_NAME}/${MODE}/${FRAG_NAME}/${SAMPLING}"

# wandb: separate local log directories by representation, model, mode, and data slice for easy identification
export WANDB_MODE=offline
export WANDB_DIR="wandb/rffmg/${MODEL_NAME}/${MODE}/${FRAG_NAME}/${SAMPLING}"
mkdir -p "${WANDB_DIR}"

if [ "$MODEL_NAME" = "t5chem" ]; then
    if [ "$MODE" = "finetuning" ]; then
        MODEL_ARG="--pretrain models/rffmg/${MODEL_NAME}/pretrained"
    else
        MODEL_ARG="--tokenizer simple"
    fi
    t5chem train ${MODEL_ARG} --data_dir data/rffmg/${FRAG_NAME}/${SAMPLING}/normal --output_dir ${OUTPUT_DIR} --task_type product --num_epoch 50

elif [ "$MODEL_NAME" = "gpt" ]; then
    # Train GPT2 (entropy/gpt2_zinc_87m) directly with transformers.
    # train_gpt.py handles finetuning/from_scratch internally.
    python src/train_model/train_gpt.py \
        --frag_method "${FRAG_NAME}" \
        --mode "${MODE}" \
        --sampling_num "${SAMPLING_NUM}"

else
    echo "Unknown MODEL_NAME: ${MODEL_NAME} (use 't5chem' or 'gpt')" >&2
    exit 1
fi
