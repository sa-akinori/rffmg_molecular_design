import argparse
import os
import sys
from glob import glob

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figure import (
    read_wandb_loss_history,
    plot_learning_curve,
    attach_points_analyze,
    dup_frags_analyze,
    dup_frags_analyze_train,
    frag_num_analyze,
)
from func.figure_func import create_boxplot, plot_single_dataset_pdf
from func.utility import BASEPATH, pickle_load

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate figures for a recursion-round rffmg t5chem run')
    parser.add_argument('--frag_method', type=str, choices=['brics', 'rc_cms'], required=True, help='fragmentation method')
    parser.add_argument('--recursion_num', type=int, required=True, help='recursion round number N (1, 2, ...)')
    args = parser.parse_args()

    frag_method = args.frag_method
    recursion_num = args.recursion_num

    scenarios = ['frag_num', 'dup_frags', 'attach_point_num']

    # (1) Performance figures (constraints) from curated_data.tsv.
    result_dir = f'{BASEPATH}/results/rffmg/t5chem/finetuning/{frag_method}/recursion/recursion{recursion_num}/beam'
    path_prefix = f'rffmg/t5chem/finetuning/{frag_method}/recursion/recursion{recursion_num}/beam'
    for const_name in scenarios:

        if const_name == 'attach_point_num':
            analyze_func = attach_points_analyze
            col_name = f'max_{const_name}'
            hue_name = 'add_frags_num'

        elif const_name == 'dup_frags':
            target_frags_path = f'{BASEPATH}/data/rffmg/{frag_method}/recursion/recursion{recursion_num}/dup_frags/target_frags.pkl'
            if not os.path.exists(target_frags_path):
                continue
            target_frags = pickle_load(target_frags_path)
            analyze_func = dup_frags_analyze(target_frags)
            col_name = 'dup_frags'
            hue_name = 'add_frags_num'

        elif const_name == 'frag_num':
            analyze_func = frag_num_analyze
            col_name = 'frag_num'
            hue_name = None

        if not os.path.exists(f'{result_dir}/{const_name}/curated_data.tsv'):
            continue

        curated_df = pd.read_csv(f'{result_dir}/{const_name}/curated_data.tsv', sep='\t', index_col=0)
        curated_df[[col_name, 'add_frags_num']] = curated_df['fragment'].apply(lambda x: pd.Series(analyze_func(x)))

        metrics = [('validratio', 'Valid ratio'), ('uniqueratio', 'Unique ratio'), ('novelratio', 'Novel ratio'), ('validfragratio', 'Validfrag ratio')]
        for metric, metric_name in metrics:
            x_lim, y_lim = [min(curated_df[col_name]) - 0.5, max(curated_df[col_name]) + 0.5], [-0.05, 1.05]
            save_path = f'{BASEPATH}/figures/constraints/{path_prefix}/{const_name}/{metric}.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            create_boxplot(df=curated_df, x_col=col_name, y_col=metric, x_name=col_name, y_name=metric_name, x_lim=x_lim, y_lim=y_lim, hue=hue_name, save_path=save_path)

    # (2) Learning curves (train/eval loss vs steps) from wandb offline runs.
    # Interrupted runs leave truncated offline-run-* siblings, so only latest-run is read.
    wandb_glob = f'{BASEPATH}/wandb/rffmg/t5chem/finetuning/{frag_method}/recursion/recursion{recursion_num}/**/latest-run/run-*.wandb'
    for run_file in sorted(glob(wandb_glob, recursive=True)):
        curve_prefix = os.path.relpath(os.path.dirname(run_file), f'{BASEPATH}/wandb').removesuffix('/wandb/latest-run')
        hist = read_wandb_loss_history(run_file)
        plot_learning_curve(hist, f'{BASEPATH}/figures/learning_curves/{curve_prefix}/curve.png', title=curve_prefix)

    # (3) Distribution of train about 'const_name'.
    data_dir = f'{BASEPATH}/data/rffmg/{frag_method}/recursion/recursion{recursion_num}'
    for const_name in scenarios:

        if const_name == 'attach_point_num':
            analyze_func = attach_points_analyze
            col_name = f'max_{const_name}'

        elif const_name == 'dup_frags':
            analyze_func = dup_frags_analyze_train
            col_name = 'dup_frags'

        elif const_name == 'frag_num':
            analyze_func = frag_num_analyze
            col_name = 'frag_num'

        if not os.path.exists(f'{data_dir}/normal/train.source'):
            continue

        train_df = pd.read_csv(f'{data_dir}/normal/train.source', sep='\t', names=['smiles'])
        train_df[[col_name, 'add_frags_num']] = train_df['smiles'].apply(lambda x: pd.Series(analyze_func(x)))
        const_count = pd.DataFrame(train_df[col_name].value_counts())
        const_count.to_csv(f'{data_dir}/{const_name}/count.csv')
        x_lim, y_lim = [min(train_df[col_name]) - 0.5, max(train_df[col_name]) + 0.5], [-0.05, train_df.shape[0] + 10]
        save_path = f'{BASEPATH}/figures/constraints/train/rffmg/{frag_method}/recursion/recursion{recursion_num}/{const_name}/train.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plot_single_dataset_pdf(data=train_df[col_name], x_label=col_name, y_label='Number of compounds', y_axis_st='float', density=False, output_path=save_path)
