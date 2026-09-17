import argparse
import ast
import itertools
import os
import random
import sys
from collections import Counter

import pandas as pd
from tqdm import tqdm

from func.utility import BASEPATH, canonical_smiles, load_file, pickle_save, save_file

# Artefact SMILES excluded from the train dedup set in make_datasets.py.
EXCLUDED_SMILES = 'O=c1/c=c\\c(=O)-n2-c3ccccc3-n-1-c1ccccc1-2'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build the recursion-round RFFMG training dataset')
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

    baseline_dir = f'{BASEPATH}/data/rffmg/{frag_method}/{sampling}'
    out_data_dir = f'{BASEPATH}/data/rffmg/{frag_method}/recursion/recursion{recursion_num}'
    normal_out_dir = f'{out_data_dir}/normal'

    # recursion1 harvests the normal-baseline results and grows the baseline train set; later
    # rounds harvest the previous round and grow the previous round's train set (cumulative).
    if recursion_num == 1:
        results_base = f'{BASEPATH}/results/rffmg/t5chem/finetuning/{frag_method}/{sampling}/beam'
        base_normal_dir = f'{baseline_dir}/normal'
    else:
        results_base = f'{BASEPATH}/results/rffmg/t5chem/finetuning/{frag_method}/recursion/recursion{recursion_num - 1}/beam'
        base_normal_dir = f'{BASEPATH}/data/rffmg/{frag_method}/recursion/recursion{recursion_num - 1}/normal'

    # (a) Harvest successful generations per scenario with a single shared RNG.
    rng = random.Random(seed)
    added_records = list()
    scenario_counts = dict()
    for scenario in scenarios:
        curated = pd.read_csv(f'{results_base}/{scenario}/curated_data.tsv', sep='\t', index_col=0)

        scenario_count = 0
        for _, row in curated.iterrows():
            if row['nvalid_onfrags'] == 0:
                continue

            smis = ast.literal_eval(row['valid_smis_on_frags'])
            picked = rng.sample(smis, min(n_select, len(smis)))
            added_records.extend([(row['fragment'], canonical_smiles(smi), scenario) for smi in picked])
            scenario_count += len(picked)

        scenario_counts[scenario] = scenario_count

    added = pd.DataFrame(added_records, columns=['source', 'target', 'scenario'])

    n_before_dedup = len(added)
    added = added.drop_duplicates(subset=['source', 'target']).reset_index(drop=True)
    n_dup_removed = n_before_dedup - len(added)

    # Molecules that entered the success set (canonical); removed from val/test below.
    SUCCESS_MOLS = set(added['target'])

    # Exclude pairs already present in the base train set.
    base_source = load_file(f'{base_normal_dir}/train.source')
    base_target = load_file(f'{base_normal_dir}/train.target')
    existing = set(zip(base_source, base_target))
    n_before_existing = len(added)
    added = added[added.apply(lambda r: (r['source'], r['target']) not in existing, axis=1)].reset_index(drop=True)
    n_existing_removed = n_before_existing - len(added)

    # (b) normal: cumulative train, and val/test with success molecules removed.
    os.makedirs(normal_out_dir, exist_ok=True)
    save_file('\n'.join(base_source + added['source'].tolist()) + '\n', f'{normal_out_dir}/train.source')
    save_file('\n'.join(base_target + added['target'].tolist()) + '\n', f'{normal_out_dir}/train.target')

    removed_counts = dict()
    for split in ['val', 'test']:
        split_source = load_file(f'{base_normal_dir}/{split}.source')
        split_target = load_file(f'{base_normal_dir}/{split}.target')
        kept_source, kept_target = list(), list()
        for source, target in zip(split_source, split_target):
            if canonical_smiles(target) in SUCCESS_MOLS:
                continue
            kept_source.append(source)
            kept_target.append(target)
        save_file('\n'.join(kept_source) + '\n', f'{normal_out_dir}/{split}.source')
        save_file('\n'.join(kept_target) + '\n', f'{normal_out_dir}/{split}.target')
        removed_counts[split] = len(split_source) - len(kept_source)

    # (c) Regenerate the three scenario test sets (make_datasets.py:175-305).
    # The fragment pool is unchanged (the molecule set does not grow), so the baseline
    # unique_frags.csv is reused; the dedup set is the expanded normal train.source.
    unique_frags_df = pd.read_csv(f'{baseline_dir}/unique_frags.csv', index_col=0)
    train_source_set = load_file(f'{normal_out_dir}/train.source')
    train_source_set = {canonical_smiles(source) for source in tqdm(train_source_set) if source != EXCLUDED_SMILES}

    # frag_num: robustness to the number of fragments in fragment sets.
    cand_frags_set = [frag for frag, count in zip(unique_frags_df['fragment'], unique_frags_df['count']) for _ in range(count)]

    val_frag_sets = list()
    for frag_num in range(1, 11):
        random.seed(frag_num)
        prev_frag_sets = list()
        while len(prev_frag_sets) < 1000:
            frag_set = '.'.join(random.sample(cand_frags_set, frag_num))
            can_frag_set = canonical_smiles(frag_set)
            if can_frag_set not in train_source_set and can_frag_set not in prev_frag_sets:
                prev_frag_sets.append(can_frag_set)
                val_frag_sets.append(frag_set)

    new_source = "\n".join(val_frag_sets) + "\n"
    new_target = "\n".join(['' for _ in val_frag_sets]) + "\n"  # T5Chem requires a target file.
    os.makedirs(f'{out_data_dir}/frag_num/', exist_ok=True)
    save_file(new_source, f'{out_data_dir}/frag_num/test.source')
    save_file(new_target, f'{out_data_dir}/frag_num/test.target')

    # dup_frags: robustness to the number of duplicated fragments in fragment sets.
    unique_frags = list(unique_frags_df['fragment'])
    frags_NHA = list(unique_frags_df['frag_NHA'])

    # The dup_num repeats of a target fragment are test prompts themselves, so they must be absent from train.
    is_unseen = lambda frag: not any(canonical_smiles('.'.join([frag] * dup_num)) in train_source_set for dup_num in range(2, 6))

    target_frag_set = list()
    for heavy_num in range(5, 21):
        target_frags = [unique_frag for unique_frag, frag_NHA in zip(unique_frags, frags_NHA) if frag_NHA == heavy_num]
        random.seed(heavy_num)
        random.shuffle(target_frags)
        target_frag_set.extend(itertools.islice(filter(is_unseen, target_frags), 30))

    cand_frags_df = unique_frags_df.query('fragment not in @target_frag_set').reset_index(drop=True)
    cand_frags_set = [frag for frag, count in zip(cand_frags_df['fragment'], cand_frags_df['count']) for _ in range(count)]

    val_frag_sets, prev_frag_sets = [['.'.join([target_frags] * dup_num) for dup_num in range(2, 6) for target_frags in target_frag_set]], list()
    for frag_num in range(1, 4):
        random.seed(frag_num)

        for dup_num in range(2, 6):
            comb_frag_sets = list()

            while len(comb_frag_sets) < 20:
                add_frags = random.sample(cand_frags_set, frag_num)
                add_frags_df = pd.DataFrame(Counter(add_frags).items(), columns=['fragment', 'count'])

                if add_frags_df.query('count > @dup_num').shape[0]:
                    add_frags_df['count'] = add_frags_df['count'].clip(upper=dup_num)

                    while (new_frag_num := frag_num - sum(add_frags_df['count'])) != 0:
                        add_frags = [frag for frag, count in zip(add_frags_df['fragment'], add_frags_df['count']) for _ in range(count)]
                        new_add_frag = random.sample(cand_frags_set, new_frag_num)
                        add_frags.extend(new_add_frag)
                        add_frags_df = pd.DataFrame(Counter(add_frags).items(), columns=['fragment', 'count'])
                        add_frags_df['count'] = add_frags_df['count'].clip(upper=dup_num)

                comb_frag_set = ['.'.join([target_frags] * dup_num + add_frags) for target_frags in target_frag_set]
                can_comb_frag_set = [canonical_smiles(frag_set) for frag_set in comb_frag_set]
                can_add_frags = canonical_smiles('.'.join(add_frags))

                if not set(can_comb_frag_set) & train_source_set and can_add_frags not in prev_frag_sets:
                    comb_frag_sets.append(comb_frag_set)
                    prev_frag_sets.append(can_add_frags)

            val_frag_sets.extend(comb_frag_sets)

    val_frag_sets = list(itertools.chain.from_iterable(val_frag_sets))
    new_source = "\n".join(val_frag_sets) + "\n"
    new_target = "\n".join(['' for _ in val_frag_sets]) + "\n"  # T5Chem requires a target file.
    os.makedirs(f'{out_data_dir}/dup_frags/', exist_ok=True)
    pickle_save(f'{out_data_dir}/dup_frags/target_frags.pkl', target_frag_set)
    save_file(new_source, f'{out_data_dir}/dup_frags/test.source')
    save_file(new_target, f'{out_data_dir}/dup_frags/test.target')

    # attach_point_num: robustness to the maximum number of attachment points in fragment sets.
    val_frag_sets, target_frag_set = list(), list()
    for max_att_point in range(2, 6):

        cond_frags_df = unique_frags_df[unique_frags_df['frag_NAP'] < max_att_point].reset_index(drop=True)
        cond_frags_set = [frag for frag, count in zip(cond_frags_df['fragment'], cond_frags_df['count']) for _ in range(count)]
        # A target fragment on its own is a test prompt itself, so fragments occurring in train are excluded.
        cand_target_df = unique_frags_df[unique_frags_df['frag_NAP'] == max_att_point]
        cand_target_df = cand_target_df[cand_target_df['fragment'].map(lambda frag: canonical_smiles(frag) not in train_source_set)]
        target_frags = list(cand_target_df.sample(n=100, replace=True, random_state=max_att_point)['fragment'])
        val_frag_sets.append(target_frags)
        target_frag_set.extend(target_frags)

        for frag_num in range(1, 4):

            frag_sets, add_frag_sets = list(), list()
            random.seed(frag_num)
            while len(frag_sets) < 100:
                add_frags = random.sample(cond_frags_set, frag_num)
                can_add_frags = canonical_smiles('.'.join(add_frags))
                frag_set = ['.'.join(random.sample(add_frags + [target_frag], len(add_frags + [target_frag]))) for target_frag in target_frags]
                can_frag_set = [canonical_smiles(rxn) for rxn in frag_set]

                if not set(can_frag_set) & train_source_set and can_add_frags not in add_frag_sets:
                    frag_sets.append(frag_set)
                    add_frag_sets.append(can_add_frags)

            val_frag_sets.extend(frag_sets)

    val_frag_sets = list(itertools.chain.from_iterable(val_frag_sets))
    new_source = "\n".join(val_frag_sets) + "\n"
    new_target = "\n".join(['' for _ in val_frag_sets]) + "\n"  # T5Chem requires a target file.
    os.makedirs(f'{out_data_dir}/attach_point_num/', exist_ok=True)
    pickle_save(f'{out_data_dir}/attach_point_num/target_frags.pkl', target_frag_set)
    save_file(new_source, f'{out_data_dir}/attach_point_num/test.source')
    save_file(new_target, f'{out_data_dir}/attach_point_num/test.target')

    # (d) Logs
    added.to_csv(f'{out_data_dir}/added_pairs.csv')

    log_lines = [
        '# Augmentation parameters',
        f'frag_method: {frag_method}',
        f'recursion_num: {recursion_num}',
        f'n_select: {n_select}',
        f'seed: {seed}',
        f'scenarios: {scenarios}',
        f'results_base: {results_base}',
        f'base_normal_dir: {base_normal_dir}',
        '',
        '# Metrics',
        *[f'picked[{scenario}]: {count}' for scenario, count in scenario_counts.items()],
        f'duplicates_removed: {n_dup_removed}',
        f'existing_in_base_removed: {n_existing_removed}',
        f'val_rows_removed: {removed_counts["val"]}',
        f'test_rows_removed: {removed_counts["test"]}',
        f'base_train_size: {len(base_source)}',
        f'final_train_size: {len(base_source) + len(added)}',
    ]
    save_file('\n'.join(log_lines) + '\n', f'{out_data_dir}/augmentation_log.txt')
