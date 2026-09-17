SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../.." || exit 1

FRAG_METHOD="rc_cms" # "brics" or "rc_cms"
RECURSION_NUM=1

~/miniconda3/envs/safe/bin/python ${SCRIPT_DIR}/figure.py --frag_method ${FRAG_METHOD} --recursion_num ${RECURSION_NUM}
