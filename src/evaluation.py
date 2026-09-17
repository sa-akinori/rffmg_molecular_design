import ast
import os
import pandas as pd
import itertools
from func.evaluation_func import *
from func.utility import *
import numpy as np
from glob import glob
import argparse

if __name__ =='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--repr_name', type=str, default='safe', choices=['rffmg', 'safe', 'promptsmiles', 'fraggpt'],
                        help='Fragment representation (method), used as the first segment of the data/, models/ '
                             'and results/ paths (default: rffmg)')
    parser.add_argument('--model_name', type=str, default='gpt', choices=['t5chem', 'gpt'],
                        help='Model the representation was trained with, used as the second segment of the '
                             'results/<repr_name>/<model_name>/ path (default: gpt)')
    parser.add_argument('--model_ver', type=str, default='finetuning', choices=['finetuning', 'pretrained', 'from_scratch'],
                        help='Phase name (default: finetuning)')
    parser.add_argument('--frag_method', type=str, default='brics', choices=['rc_cms', 'brics'],
                        help='Fragmentation method (default: rc_cms)')
    parser.add_argument('--additional_path', type=str, default='normal', choices=['normal', 'dup_frags', 'frag_num', 'frag_order', 'attach_point_num'],
                        help='Additional path (default: normal)')
    parser.add_argument('--gen_method', type=str, default='beam', choices=['beam', 'sampling'],
                        help='Decoding scheme segment of the results path; defaults to the one the model was generated with')
    parser.add_argument('--sampling_num', type=int, default=5, choices=[5, 10],
                        help='Number of fragmentation patterns per molecule, used as the data/rffmg/<frag_method>/<N>times_sampling '
                             'segment of the RFFMG data and results paths (default: 5). Ignored unless --repr_name is rffmg')
    args = parser.parse_args()
    if args.model_name == 't5chem' and args.repr_name != 'rffmg':
        parser.error('--model_name t5chem is only available with --repr_name rffmg')

    # Setting
    repr_name   = args.repr_name
    model_name  = args.model_name
    gen_method  = args.gen_method
    model_ver   = args.model_ver
    frag_method = args.frag_method
    additional_path = args.additional_path
    sampling     = f'{args.sampling_num}times_sampling'
    sampling_seg = f'{sampling}/' if repr_name == 'rffmg' else ''
    cpu_num = os.cpu_count()

    if 0:
        # Load dataset
        if repr_name == 'rffmg':
            tr_file_name  = f'{BASEPATH}/data/rffmg/{frag_method}/{sampling}/normal/train.target'
            testInputfile = f'{BASEPATH}/data/rffmg/{frag_method}/{sampling}/{additional_path}/test.source'

        else:  # safe, promptsmiles or fraggpt
            tr_file_name  = f'{BASEPATH}/data/{repr_name}/{frag_method}/normal'
            testInputfile = None

        trsmiles = loadTrainSmiles(tr_file_name)
        
        # Calculate some basic physic property for training smiles
        if repr_name == 'rffmg':
            if not os.path.isfile(f'{BASEPATH}/results/physical_properties/train/{sampling}/physical_property.csv'):
                trPhysicprop = calcPhysicProp(list(trsmiles), n_jobs=cpu_num-1)
                trPhysicprop_df = pd.DataFrame(trPhysicprop)
                os.makedirs(f'{BASEPATH}/results/physical_properties/train/{sampling}', exist_ok=True)
                trPhysicprop_df.to_csv(f'{BASEPATH}/results/physical_properties/train/{sampling}/train_physical_property.csv')
            
        
        outfd = f'{BASEPATH}/results/{repr_name}/{model_name}/{model_ver}/{frag_method}/{sampling_seg}{gen_method}/{additional_path}'

        ## Calculate physical property
        # Define file path
        file_name = f'{outfd}/predictions.csv'
        
        # Evaluation gen mols
        genmols = pd.read_csv(file_name)

        if repr_name == 'rffmg':
            inmols  = pd.read_csv(testInputfile, sep='>', header=None, names=['fragment']).iloc[:,[0]]
            genmols = pd.concat([inmols, genmols], axis=1)

        pred_cols = [col for col in genmols.columns if col.startswith('prediction_')]
        genmols = genmols[['fragment', 'target'] + pred_cols]
        stats, genmols = sc3_check_genmol_results(outfd=outfd, genmols=genmols, trsmiles=trsmiles, skipCreateExcel=True, algorithm_name=frag_method, n_chunks=5)
        stats.to_csv(f'{outfd}/stats.csv')
        
        # Calculate some basic physic property for training smiles
        # novel_smi comes back from the chunk TSVs as the repr of a set, so it has to be parsed before
        # it can be iterated over. An empty set is written as 'set()', which ast.literal_eval rejects.
        parse_smiles_set = lambda text: set() if text == 'set()' else ast.literal_eval(text)
        gensmiles = list({smi for _, row in genmols.iterrows() for smi in parse_smiles_set(row['novel_smi'])})
        genPhysicprop = calcPhysicProp(list(gensmiles), n_jobs=cpu_num-1)
        genPhysicprop_df = pd.DataFrame(genPhysicprop)
        genPhysicprop_df.to_csv(f'{outfd}/physical_property.csv')

        # Evaluation metrics only for fragments used in frag_order (to compare the performance between unshuffled and shuffled fragment orders)
        if additional_path == 'frag_order':
            
            outfd  = f'{BASEPATH}/results/{repr_name}/{model_name}/{model_ver}/{frag_method}/{sampling_seg}{gen_method}'
            datafd = f'{BASEPATH}/data/rffmg/{frag_method}/{sampling}'
            no_shuffle_df = pd.read_csv(f'{outfd}/normal/curated_data.tsv', sep='\t', index_col=0)
            random_get_id = pickle_load(f'{datafd}/frag_order/random_get_ids.pkl')
            no_shuffle_df = no_shuffle_df.loc[random_get_id]
            
            stats = dict()
            stats['avg_validity']         = no_shuffle_df['validratio'].mean() 
            stats['std_validity']         = no_shuffle_df['validratio'].std() 
            stats['avg_validity_onfrags'] = no_shuffle_df['validfragratio'].mean() # unique fragments should be used and count should be reflected.
            stats['std_validity_onfrags'] = no_shuffle_df['validfragratio'].std()
            stats['avg_uniqueness']       = no_shuffle_df['uniqueratio'].mean()
            stats['std_uniqueness']       = no_shuffle_df['uniqueratio'].std()
            stats['avg_novelty']          = no_shuffle_df['novelratio'].mean()
            stats['std_novelty']          = no_shuffle_df['novelratio'].std()
            # 
            stats['avg_tanimoto_sim']     = no_shuffle_df[no_shuffle_df['nnovel'] != 0]['tanimoto_sim'].mean()
            stats['std_tanimoto_sim']     = no_shuffle_df[no_shuffle_df['nnovel'] != 0]['tanimoto_sim'].std()
            stats['avg_tanimoto_sim_onfrags'] = no_shuffle_df[no_shuffle_df['nvalid_onfrags'] != 0]['tanimoto_sim_onfrags'].mean()
            stats['std_tanimoto_sim_onfrags'] = no_shuffle_df[no_shuffle_df['nvalid_onfrags'] != 0]['tanimoto_sim_onfrags'].std()
            
            stats_df = pd.Series(stats)
            stats_df.to_csv(f'{outfd}/frag_order/no_shuffle_stats.csv')
        
    if 1: # Calculate js-divergence between train and test
        # Setting
        compared_files = [pd.read_csv(file, index_col=0) for file in [
            f'{BASEPATH}/results/train_physical_property.csv',
            f'{BASEPATH}/results/rffmg/t5chem/finetuning/{frag_method}/{args.sampling_num}times_sampling/beam/normal/physical_property.csv',
            f'{BASEPATH}/results/rffmg/gpt/finetuning/{frag_method}/{args.sampling_num}times_sampling/beam/normal/physical_property.csv',
            f'{BASEPATH}/results/safe/gpt/pretrained/{frag_method}/beam/normal/physical_property.csv',
            f'{BASEPATH}/results/safe/gpt/finetuning/{frag_method}/beam/normal/physical_property.csv']
            ]
        properties = ['MW', 'TPSA', 'LogP', 'QED']
        bin_sizes  = [1, 1, 0.1, 0.01]
        file_names = ['train', 't5chem_trained_rffmg', 'gpt_trained_rffmg', 'gpt_pretrained_safe', 'gpt_trained_safe']
        
        for prop_name, bin_size in zip(properties, bin_sizes):
            
            # Calculate js-divergence
            js_div = calculate_js_divergence_for_properties(prop_dfs=compared_files, file_names=file_names, prop_name=prop_name, bin_size=bin_size)
            os.makedirs(f'{BASEPATH}/results/js_divergence/physical_properties/{frag_method}/beam/normal/', exist_ok=True)
            js_div.to_csv(f'{BASEPATH}/results/js_divergence/physical_properties/{frag_method}/beam/normal/{prop_name}.csv')

    
    if 0: # Evaluation metrics split by how much freedom the fragment set leaves for the connection:
        
        if additional_path == 'normal':

            outfd = f'{BASEPATH}/results/{repr_name}/{model_name}/{model_ver}/{frag_method}/{sampling_seg}{gen_method}/{additional_path}'
            curated_data = pd.read_csv(f'{outfd}/curated_data.tsv', sep='\t', index_col=0)
            classify_constraint = lambda frags: 'DC' if len(frags.split('.')) == 1 or max(frag.count('*') for frag in frags.split('.')) <= 1 else 'OC'
            curated_data['constraint'] = curated_data['fragment'].map(classify_constraint)

            # A target found anywhere in the candidate list counts as rediscovered; rank 0 means not found.
            curated_data['rediscovered'] = (curated_data['rank'] != 0).astype(int)

            dc_oc_stats = {'n_samples': curated_data['constraint'].value_counts()}
            for name, col in {'validity': 'validratio', 'validity_onfrags': 'validfragratio', 'uniqueness': 'uniqueratio', 'novelty': 'novelratio', 'rediscovery': 'rediscovered'}.items():
                grouped = curated_data.groupby('constraint')[col]
                dc_oc_stats[f'avg_{name}'], dc_oc_stats[f'std_{name}'] = grouped.mean(), grouped.std()

            # The similarities are averaged over the rows that produced the molecules they compare.
            for name, count_col in {'tanimoto_sim': 'nnovel', 'tanimoto_sim_onfrags': 'nvalid_onfrags'}.items():
                grouped = curated_data[curated_data[count_col] != 0].groupby('constraint')[name]
                dc_oc_stats[f'avg_{name}'], dc_oc_stats[f'std_{name}'] = grouped.mean(), grouped.std()

            pd.DataFrame(dc_oc_stats).T.to_csv(f'{outfd}/dc_oc_stats.csv')
