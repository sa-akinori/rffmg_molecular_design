#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# リポジトリルートに移動
cd "${SCRIPT_DIR}/../.." || exit 1

FRAG_METHOD="brics" # "brics" or "rc_cms"
RECURSION_NUM=1
ADDITIONAL_PATH="normal" # normal / frag_num / dup_frags / attach_point_num

~/miniconda3/envs/safe/bin/python ${SCRIPT_DIR}/evaluate.py --frag_method ${FRAG_METHOD} --recursion_num ${RECURSION_NUM} --additional_path ${ADDITIONAL_PATH}
