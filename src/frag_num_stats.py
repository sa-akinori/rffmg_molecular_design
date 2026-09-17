import argparse

import pandas as pd

from func.utility import BASEPATH

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tabulate the fragment-count distribution of the RFFMG train split')
    parser.add_argument('--frag_method', type=str, choices=['brics', 'rc_cms'], required=True, help='fragmentation method')
    parser.add_argument('--sampling_num', type=int, required=True, help='number of sampling trials per fragment set (data/rffmg/<frag>/<N>times_sampling)')
    args = parser.parse_args()

    normal_dir = f'{BASEPATH}/data/rffmg/{args.frag_method}/{args.sampling_num}times_sampling/normal'

    train_df = pd.read_csv(f'{normal_dir}/train.source', sep='\t', names=['fragment'])
    counts = train_df['fragment'].str.split('.').str.len().value_counts().sort_index()

    stats_df = pd.DataFrame({'count': counts, 'ratio': counts / len(train_df)})
    stats_df.index.name = 'frag_num'
    stats_df.to_csv(f'{normal_dir}/frag_num_stats.csv')
