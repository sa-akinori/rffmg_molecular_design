import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from func.utility import BASEPATH

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate molecules with a recursion-round rffmg t5chem model (one condition)')
    parser.add_argument('--frag_method', type=str, choices=['rc_cms', 'brics'], required=True, help='fragmentation method')
    parser.add_argument('--recursion_num', type=int, required=True, help='recursion round number N (1, 2, ...)')
    parser.add_argument('--additional_path', type=str, choices=['normal', 'frag_num', 'dup_frags', 'attach_point_num'], required=True,
                        help='generation condition (test dataset slice)')
    parser.add_argument('--num_beams', type=int, default=50, help='number of beams for beam search (default: 50)')
    parser.add_argument('--n_samples', type=int, default=50, help='number of samples to generate per molecule (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size (default: 8)')
    args = parser.parse_args()

    frag_method = args.frag_method
    recursion_num = args.recursion_num
    additional_path = args.additional_path

    model_dir = f'{BASEPATH}/models/rffmg/t5chem/finetuning/{frag_method}/recursion/recursion{recursion_num}/best_model'
    data_dir = f'{BASEPATH}/data/rffmg/{frag_method}/recursion/recursion{recursion_num}/{additional_path}'
    output_dir = f'{BASEPATH}/results/rffmg/t5chem/finetuning/{frag_method}/recursion/recursion{recursion_num}/beam/{additional_path}'
    os.makedirs(output_dir, exist_ok=True)

    subprocess.run(
        [
            't5chem', 'predict',
            '--data_dir', f'{data_dir}/',
            '--model_dir', f'{model_dir}/',
            '--prediction', f'{output_dir}/predictions.csv',
            '--num_beams', str(args.num_beams),
            '--num_preds', str(args.n_samples),
            '--batch_size', str(args.batch_size),
        ],
        check=True,
    )
