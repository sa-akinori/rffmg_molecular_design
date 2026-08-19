import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from func.utility import BASEPATH

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune t5chem on the recursion-round rffmg dataset')
    parser.add_argument('--frag_method', type=str, choices=['brics', 'rc_cms'], required=True, help='fragmentation method')
    parser.add_argument('--recursion_num', type=int, required=True, help='recursion round number N (1, 2, ...)')
    args = parser.parse_args()

    frag_method = args.frag_method
    recursion_num = args.recursion_num

    data_dir = f'{BASEPATH}/data/rffmg/{frag_method}/recursion/recursion{recursion_num}/normal'
    output_dir = f'{BASEPATH}/models/rffmg/t5chem/finetuning/{frag_method}/recursion/recursion{recursion_num}'

    subprocess.run(
        [
            't5chem', 'train',
            '--pretrain', f'{BASEPATH}/models/rffmg/t5chem/pretrained',
            '--data_dir', data_dir,
            '--output_dir', output_dir,
            '--task_type', 'product',
            '--num_epoch', '50',
        ],
        check=True,
    )
