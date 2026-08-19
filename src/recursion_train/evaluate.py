import argparse
import ast
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from func.evaluation_func import sc3_check_genmol_results, loadTrainSmiles, calcPhysicProp
from func.utility import BASEPATH

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate a recursion-round rffmg t5chem generation (one condition)')
    parser.add_argument('--frag_method', type=str, choices=['rc_cms', 'brics'], required=True, help='fragmentation method')
    parser.add_argument('--recursion_num', type=int, required=True, help='recursion round number N (1, 2, ...)')
    parser.add_argument('--additional_path', type=str, choices=['normal', 'frag_num', 'dup_frags', 'attach_point_num'], required=True,
                        help='generation condition (test dataset slice)')
    args = parser.parse_args()

    frag_method = args.frag_method
    recursion_num = args.recursion_num
    additional_path = args.additional_path
    cpu_num = os.cpu_count()

    outfd = f'{BASEPATH}/results/rffmg/t5chem/finetuning/{frag_method}/recursion/recursion{recursion_num}/beam/{additional_path}'
    testInputfile = f'{BASEPATH}/data/rffmg/{frag_method}/5times_sampling/{additional_path}/test.source'

    # Novelty is measured against the round's expanded train set.
    tr_file = f'{BASEPATH}/data/rffmg/{frag_method}/recursion/recursion{recursion_num}/train.target'
    trsmiles = loadTrainSmiles(tr_file)

    genmols = pd.read_csv(f'{outfd}/predictions.csv')
    inmols = pd.read_csv(testInputfile, sep='>', header=None, names=['fragment']).iloc[:, [0]]
    genmols = pd.concat([inmols, genmols], axis=1)

    pred_cols = [col for col in genmols.columns if col.startswith('prediction_')]
    genmols = genmols[['fragment', 'target'] + pred_cols]
    stats, genmols = sc3_check_genmol_results(outfd=outfd, genmols=genmols, trsmiles=trsmiles, skipCreateExcel=True, algorithm_name=frag_method, n_chunks=5)
    stats.to_csv(f'{outfd}/stats.csv')

    # novel_smi comes back from the chunk TSVs as the repr of a set, so it has to be parsed before
    # it can be iterated over. An empty set is written as 'set()', which ast.literal_eval rejects.
    parse_smiles_set = lambda text: set() if text == 'set()' else ast.literal_eval(text)
    gensmiles = list({smi for _, row in genmols.iterrows() for smi in parse_smiles_set(row['novel_smi'])})
    genPhysicprop = calcPhysicProp(list(gensmiles), n_jobs=cpu_num - 1)
    genPhysicprop_df = pd.DataFrame(genPhysicprop)
    genPhysicprop_df.to_csv(f'{outfd}/physic_property.csv')
