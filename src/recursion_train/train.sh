#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# リポジトリルートに移動（相対パスを解決するため）
cd "$(cd "$(dirname "$0")" && pwd)/../.." || exit 1

# conda setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate t5chem

FRAG_METHOD="brics" # "brics" or "rc_cms"
RECURSION_NUM=1

DATA_DIR="data/rffmg/${FRAG_METHOD}/recursion/recursion${RECURSION_NUM}/normal"
OUTPUT_DIR="models/rffmg/t5chem/finetuning/${FRAG_METHOD}/recursion/recursion${RECURSION_NUM}"

# wandb: オフライン + recursion スライスごとに保存先を分ける
export WANDB_MODE=offline
export WANDB_DIR="wandb/rffmg/t5chem/finetuning/${FRAG_METHOD}/recursion/recursion${RECURSION_NUM}"
mkdir -p "${WANDB_DIR}"

t5chem train --pretrain models/rffmg/t5chem/pretrained --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --task_type product --num_epoch 50
