SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rffmg

FRAG_METHOD="brics" # "brics" or "rc_cms"
RECURSION_NUM=1
ADDITIONAL_PATH="normal" # normal / frag_num / dup_frags / attach_point_num

python ${SCRIPT_DIR}/generate.py --frag_method ${FRAG_METHOD} --recursion_num ${RECURSION_NUM} --additional_path ${ADDITIONAL_PATH}
