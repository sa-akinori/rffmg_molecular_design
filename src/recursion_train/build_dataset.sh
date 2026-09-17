SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../.." || exit 1

# conda setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate t5chem

FRAG_METHOD="brics" # "brics" or "rc_cms"
RECURSION_NUM=1

python ${SCRIPT_DIR}/build_dataset.py --frag_method ${FRAG_METHOD} --recursion_num ${RECURSION_NUM}
