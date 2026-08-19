import argparse
import ast
import os
import random
import shutil
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from func.utility import BASEPATH, load_file, save_file

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build the recursion-round RFFMG training dataset and finetune t5chem on it')
    parser.add_argument('--frag_method', type=str, choices=['brics', 'rc_cms'], required=True, help='fragmentation method')
    parser.add_argument('--recursion_num', type=int, required=True, help='recursion round number N (1, 2, ...)')
    parser.add_argument('--n_select', type=int, default=5, help='number of successful molecules to keep per fragment set (default: 5)')
    parser.add_argument('--seed', type=int, default=0, help='random seed for molecule sampling (default: 0)')
    args = parser.parse_args()

    frag_method = args.frag_method
    recursion_num = args.recursion_num
    n_select = args.n_select
    seed = args.seed

    scenarios = ['frag_num', 'dup_frags', 'attach_point_num']
    sampling = '5times_sampling'

    normal_dir = f'{BASEPATH}/data/rffmg/{frag_method}/{sampling}/normal'
    out_data_dir = f'{BASEPATH}/data/rffmg/{frag_method}/recursion/recursion{recursion_num}'

    # recursion1 harvests the normal-baseline results and grows the normal train set; later rounds
    # harvest the previous round and grow the previous round's train set (cumulative).
    if recursion_num == 1:
        results_base = f'{BASEPATH}/results/rffmg/t5chem/finetuning/{frag_method}/{sampling}/beam'
        base_train_dir = normal_dir
    else:
        results_base = f'{BASEPATH}/results/rffmg/t5chem/finetuning/{frag_method}/recursion/recursion{recursion_num - 1}/beam'
        base_train_dir = f'{BASEPATH}/data/rffmg/{frag_method}/recursion/recursion{recursion_num - 1}'

    # Harvest successful generations per scenario
    rng = random.Random(seed)
    added_records = list()
    scenario_counts = dict()
    for scenario in scenarios:
        curated = pd.read_csv(f'{results_base}/{scenario}/curated_data.tsv', sep='\t', index_col=0)

        scenario_count = 0
        for _, row in curated.iterrows():
            # Skip fragment sets with no valid generation
            if row['nvalid_onfrags'] == 0:
                continue

            smis = ast.literal_eval(row['valid_smis_on_frags'])
            picked = rng.sample(smis, min(n_select, len(smis)))
            added_records.extend([(row['fragment'], smi, scenario) for smi in picked])
            scenario_count += len(picked)

        scenario_counts[scenario] = scenario_count

    added = pd.DataFrame(added_records, columns=['source', 'target', 'scenario'])

    # Remove exact duplicates among the harvested pairs
    n_before_dedup = len(added)
    added = added.drop_duplicates(subset=['source', 'target']).reset_index(drop=True)
    n_dup_removed = n_before_dedup - len(added)

    # Exclude pairs already present in the base train set
    base_source = load_file(f'{base_train_dir}/train.source')
    base_target = load_file(f'{base_train_dir}/train.target')
    existing = set(zip(base_source, base_target))
    n_before_existing = len(added)
    added = added[added.apply(lambda r: (r['source'], r['target']) not in existing, axis=1)].reset_index(drop=True)
    n_existing_removed = n_before_existing - len(added)

    os.makedirs(out_data_dir, exist_ok=True)

    # Save cumulative train (base train + harvested pairs)
    save_file('\n'.join(base_source + added['source'].tolist()) + '\n', f'{out_data_dir}/train.source')
    save_file('\n'.join(base_target + added['target'].tolist()) + '\n', f'{out_data_dir}/train.target')

    # Copy val/test from normal unchanged
    for split in ['val', 'test']:
        for ext in ['source', 'target']:
            shutil.copyfile(f'{normal_dir}/{split}.{ext}', f'{out_data_dir}/{split}.{ext}')

    added.to_csv(f'{out_data_dir}/added_pairs.csv')

    n_final_train = len(base_source) + len(added)
    log_lines = [
        '# Augmentation parameters',
        f'frag_method: {frag_method}',
        f'recursion_num: {recursion_num}',
        f'n_select: {n_select}',
        f'seed: {seed}',
        f'scenarios: {scenarios}',
        f'results_base: {results_base}',
        f'base_train_dir: {base_train_dir}',
        '',
        '# Metrics',
        *[f'picked[{scenario}]: {count}' for scenario, count in scenario_counts.items()],
        f'duplicates_removed: {n_dup_removed}',
        f'existing_in_base_removed: {n_existing_removed}',
        f'base_train_size: {len(base_source)}',
        f'final_train_size: {n_final_train}',
    ]
    save_file('\n'.join(log_lines) + '\n', f'{out_data_dir}/augmentation_log.txt')

    # Finetune t5chem on the recursion-round dataset (same settings as train_model/run_rffmg.sh)
    subprocess.run(
        [
            't5chem', 'train',
            '--pretrain', f'{BASEPATH}/models/rffmg/t5chem/pretrained',
            '--data_dir', out_data_dir,
            '--output_dir', f'{BASEPATH}/models/rffmg/t5chem/finetuning/{frag_method}/recursion/recursion{recursion_num}',
            '--task_type', 'product',
            '--num_epoch', '50',
        ],
        check=True,
    )
