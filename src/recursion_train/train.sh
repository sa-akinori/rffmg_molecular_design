#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# リポジトリルートに移動（t5chem CLI が相対パスを解決できるようにする）
cd "${SCRIPT_DIR}/../.." || exit 1

# conda setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate t5chem

FRAG_METHOD="brics" # "brics" or "rc_cms"
RECURSION_NUM=1

python ${SCRIPT_DIR}/train.py --frag_method ${FRAG_METHOD} --recursion_num ${RECURSION_NUM}
