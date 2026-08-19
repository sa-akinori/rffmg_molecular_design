# Plan: recursion_train パイプライン(データ作成+学習 / 生成 / 評価)

- **Date**: 2026-08-19
- **Status**: pending-approval

## Overview

ロバスト性評価シナリオ(`frag_num` / `dup_frags` / `attach_point_num`)で **生成に成功した分子**(`curated_data.tsv` の `valid_smis_on_frags`)を訓練データに累積追加し、rffmg × t5chem(finetuning)モデルを再学習して、再度4条件(`normal` + 3シナリオ)で生成精度を検証する「再帰的自己学習(recursion training)」パイプラインを `src/recursion_train/` に構築する。

**スコープ / 制約**:
- 対象は **rffmg × t5chem × finetuning** のみ。frag_method はファイル先頭変数(既定 `brics`)。`sampling` は `5times_sampling` 固定(パラメータ化しない)。
- ラベル `recursion{N}`(N=1,2,…)を各スクリプトに渡す。
- **`src/recursion_train/` 以外の `src/` は一切変更しない**。既存コードは subprocess 起動 or import で再利用する。
- 既存 `src/recursion_train/build_augmented_dataset.py` / `.sh` は **削除**して新規作成する。
- 既存の汎用関数(`func.utility.BASEPATH`、`make_datasets.load_file/save_file`、`func.evaluation_func.*`)は再利用する(重複実装を作らない)。

**出力パス**(`{frag}` は frag_method、`{N}` は recursion 番号。`{sampling}` は入れない):
- data: `data/rffmg/{frag}/recursion/recursion{N}/`(train=累積拡張、val/test=normal からコピー)
- model: `models/rffmg/t5chem/finetuning/{frag}/recursion/recursion{N}/best_model`
- results: `results/rffmg/t5chem/finetuning/{frag}/recursion/recursion{N}/beam/{scenario}/`

**成功例の収穫元**(`curated_data.tsv`):
- recursion1 → normal baseline: `results/rffmg/t5chem/finetuning/{frag}/5times_sampling/beam/{scenario}/curated_data.tsv`
- recursion{N≥2} → 直前ラウンド: `results/rffmg/t5chem/finetuning/{frag}/recursion/recursion{N-1}/beam/{scenario}/curated_data.tsv`

**運用フロー**(各段は手動、ドライバなし):
`(normal baseline を評価済み)` → build_and_train r1 → generate r1 ×4条件 → evaluate r1 ×4条件 → build_and_train r2(r1 の curated を収穫)→ …

**スクリプト構成(3ステージ、各 `.py`+`.sh`)**:
1. build_and_train(env `t5chem`): データ作成と学習を統合(1ラウンド分)。
2. generate(env `t5chem`): 1条件ずつ生成。
3. evaluate(env `safe`): 1条件ずつ評価。

## Plan

### Step 1: 旧 build_augmented_dataset を削除

- **Target file**: `src/recursion_train/build_augmented_dataset.py`, `src/recursion_train/build_augmented_dataset.sh`
- **Changes**: 両ファイルを削除(`git rm`)。`__pycache__` の残骸も除去。
- **Dependencies**: none

### Step 2: build_and_train.py(データ作成 + 学習 統合)

- **Target file**: `src/recursion_train/build_and_train.py`(新規)
- **Changes**:
  - `argparse`: `--frag_method {brics,rc_cms}`(required)、`--recursion_num`(int, required)、`--n_select`(int, 既定 5)、`--seed`(int, 既定 0)。`scenarios` は `['frag_num','dup_frags','attach_point_num']` 固定、`sampling='5times_sampling'` / `model_ver='finetuning'` 固定。
  - パス基盤に `func.utility.BASEPATH`、ファイル IO に `make_datasets.load_file/save_file` を import して再利用(`sys.path` に `src/` を追加)。
  - **データ作成**:
    1. 収穫元 results dir を決定(N==1 は 5times_sampling/beam、N≥2 は recursion/recursion{N-1}/beam)。
    2. base 訓練データを決定(N==1 は `data/rffmg/{frag}/5times_sampling/normal/`、N≥2 は `data/rffmg/{frag}/recursion/recursion{N-1}/`)。
    3. 各シナリオの `curated_data.tsv` を `pd.read_csv(sep='\t', index_col=0)` で読み、`nvalid_onfrags==0` 行はスキップ。`ast.literal_eval(row['valid_smis_on_frags'])` から `random.Random(seed).sample(smis, min(n_select, len(smis)))` を抽選し `(fragment, smi)` を蓄積。
    4. `added` DataFrame(列 `source,target,scenario`)を作り、`(source,target)` で dedup。さらに base train の `(source,target)` 集合に既存のものを除外(二重 dedup)。
    5. 出力 `data/rffmg/{frag}/recursion/recursion{N}/` に、`train.source/target` = base train + added(**累積**)を書き出し。`val.*` / `test.*` は normal(`data/rffmg/{frag}/5times_sampling/normal/`)からコピー(`shutil.copyfile`)。`added_pairs.csv` と `augmentation_log.txt`(picked 件数・dedup 件数・train サイズ等のメトリクス)も出力。
  - **学習**(データ作成成功後に続けて実行):
    - `subprocess.run` で `t5chem train --pretrain {BASEPATH}/models/rffmg/t5chem/pretrained --data_dir {BASEPATH}/data/rffmg/{frag}/recursion/recursion{N} --output_dir {BASEPATH}/models/rffmg/t5chem/finetuning/{frag}/recursion/recursion{N} --task_type product --num_epoch 50` を起動(既存 `src/train_model/run_rffmg.sh:29-35` の t5chem finetuning 分岐と同一設定)。`check=True`。
- **Dependencies**: after Step 1

### Step 3: build_and_train.sh(t5chem 環境ラッパー)

- **Target file**: `src/recursion_train/build_and_train.sh`(新規)
- **Changes**:
  - 既存 `.sh` 流儀: `SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"`、`source ~/miniconda3/etc/profile.d/conda.sh` → `conda activate t5chem`(`run_rffmg.sh:8-9` と同じ)。cwd をプロジェクトルートに移動(t5chem CLI の相対パス基準)。
  - ファイル先頭変数: `FRAG_METHOD="brics" # "brics" or "rc_cms"`、`RECURSION_NUM=1`、`N_SELECT=5`、`SEED=0`。
  - `python ${SCRIPT_DIR}/build_and_train.py --frag_method ${FRAG_METHOD} --recursion_num ${RECURSION_NUM} --n_select ${N_SELECT} --seed ${SEED}` を実行。
- **Dependencies**: after Step 2

### Step 4: generate.py(1条件生成)

- **Target file**: `src/recursion_train/generate.py`(新規)
- **Changes**:
  - `argparse`: `--frag_method`(required)、`--recursion_num`(int, required)、`--additional_path {normal,frag_num,dup_frags,attach_point_num}`(required)、生成パラメータ `--num_beams`(既定 50)、`--n_samples`(既定 50)、`--batch_size`(既定 8)、`--max_length`(既定 256)。
  - パス構築:
    - `model_dir = {BASEPATH}/models/rffmg/t5chem/finetuning/{frag}/recursion/recursion{N}/best_model`
    - `data_dir  = {BASEPATH}/data/rffmg/{frag}/5times_sampling/{additional_path}`(既存シナリオ test。変更しない)
    - `output_dir= {BASEPATH}/results/rffmg/t5chem/finetuning/{frag}/recursion/recursion{N}/beam/{additional_path}`(`os.makedirs(exist_ok=True)`)
  - `src/gen_mols/gen_rffmg.py` の **t5chem predict 呼び出し(該当行)を忠実に再現**して `subprocess.run` で `t5chem predict` を起動し `{output_dir}/predictions.csv` を生成(フラグ名・引数は gen_rffmg.py に合わせる)。時間計測(`run_and_record_time`)は不要のため省く。
- **Dependencies**: after Step 3(モデルが必要)

### Step 5: generate.sh(t5chem 環境ラッパー)

- **Target file**: `src/recursion_train/generate.sh`(新規)
- **Changes**:
  - `conda activate t5chem`(`gen_rffmg.sh:5-6` と同じ)。cwd をプロジェクトルートへ。
  - 先頭変数: `FRAG_METHOD="brics"`、`RECURSION_NUM=1`、`ADDITIONAL_PATH="normal" # normal / frag_num / dup_frags / attach_point_num`(1条件ずつ。4条件は変数を変えて4回実行)。
  - `python ${SCRIPT_DIR}/generate.py --frag_method ${FRAG_METHOD} --recursion_num ${RECURSION_NUM} --additional_path ${ADDITIONAL_PATH}` を実行。
- **Dependencies**: after Step 4

### Step 6: evaluate.py(1条件評価)

- **Target file**: `src/recursion_train/evaluate.py`(新規)
- **Changes**:
  - `argparse`: `--frag_method`(required)、`--recursion_num`(int, required)、`--additional_path {normal,frag_num,dup_frags,attach_point_num}`(required)。
  - `func.evaluation_func` から `sc3_check_genmol_results`, `loadTrainSmiles`, `calcPhysicProp` を、`func.utility` から `BASEPATH` を import。
  - パス:
    - `outfd = {BASEPATH}/results/rffmg/t5chem/finetuning/{frag}/recursion/recursion{N}/beam/{additional_path}`
    - predictions = `{outfd}/predictions.csv`
    - test.source = `{BASEPATH}/data/rffmg/{frag}/5times_sampling/{additional_path}/test.source`
    - **novelty 基準 train = 拡張後 train**: `tr_file = {BASEPATH}/data/rffmg/{frag}/recursion/recursion{N}/train.target` を `loadTrainSmiles` で `trsmiles` に。
  - `src/evaluation.py __main__` の rffmg 向けグルー(predictions.csv 読込 → test.source を **行番号 join** して `fragment` 列付与 → `all_smiles` リスト列と `['fragment','target',prediction_*]` の genmols を構築 → `sc3_check_genmol_results(outfd=outfd, genmols=genmols, trsmiles=trsmiles, skipCreateExcel=True, algorithm_name=frag_method, n_chunks=5)` 呼出 → `stats.to_csv(f'{outfd}/stats.csv')` → novel SMILES に `calcPhysicProp` して `physic_property.csv` 出力)を **該当行を参照して忠実に再現**(約15行、evaluation.py を編集せず複製)。`curated_data.tsv` は `sc3_check_genmol_results` が出力。
- **Dependencies**: after Step 5(predictions が必要)

### Step 7: evaluate.sh(safe 環境ラッパー)

- **Target file**: `src/recursion_train/evaluate.sh`(新規)
- **Changes**:
  - 評価は `datasets` 依存のため **safe 環境**を使う(`run_cpu.sh:6` と同様、`~/miniconda3/envs/safe/bin/python` を直接呼ぶ、または `conda activate safe`)。cwd をプロジェクトルートへ。
  - 先頭変数: `FRAG_METHOD="brics"`、`RECURSION_NUM=1`、`ADDITIONAL_PATH="normal"`(1条件ずつ、4回実行)。
  - `evaluate.py` を上記引数で実行。
- **Dependencies**: after Step 6

## Notes / 前提

- recursion1 を回す前に、**normal baseline(t5chem finetuning)の3シナリオ評価**(`curated_data.tsv`)が済んでいること。これは既存パイプラインの成果物であり本計画の対象外。
- コードスタイル: 型ヒント必須、Google style docstring、import 順(標準→サードパーティ→ローカル)、単純変換は lambda、無駄なコメント/抽象化を作らない、乱数 seed 明示、SMILES バリデーション省略しない。
- 実装は承認後 `implementer` に委譲する。main ブランチへ直接コミット(worktree・作業ブランチは作らない)。
