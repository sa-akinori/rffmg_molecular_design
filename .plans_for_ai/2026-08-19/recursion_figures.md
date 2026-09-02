# Plan: recursion_train の図生成スクリプト

- **Date**: 2026-08-19
- **Status**: pending-approval

## Overview

recursion ラウンドの結果から図を生成する `src/recursion_train/figure.py`(+ `figure.sh`)を新規作成する。既存 `src/figure.py` は固定グリッド(`{sampling}times_sampling`)前提で recursion ツリー(`.../recursion/recursion{N}/...`)に到達できないため、該当ブロックを複製し、パスを recursion 用に差し替える。生成する図は3種:

1. **性能図(constraints)**: `frag_num` / `dup_frags` / `attach_point_num` の各シナリオについて、`curated_data.tsv` から `validratio` / `uniqueratio` / `novelratio` / `validfragratio` を制約軸に対してボックスプロット(`src/figure.py` 325-366 相当)。
2. **学習曲線**: 当該 recursion ラウンドの wandb オフラインログから train/eval loss(`src/figure.py` 428-434 相当)。
3. **訓練データの分布**: `recursion{N}/normal/train.source` 内の `frag_num` / `dup_frags` / `attach_point_num` 各制約値ごとの分子数ヒストグラム(`src/figure.py` 368-405 相当)。

**再利用**(`src/` は変更せず import):
- `src/figure.py` から `read_wandb_loss_history`, `plot_learning_curve`, `attach_points_analyze`, `dup_frags_analyze`, `dup_frags_analyze_train`, `frag_num_analyze`(いずれも module 直下・副作用なし)。
- `func.figure_func` から `create_boxplot`, `plot_single_dataset_pdf`。
- `func.utility` から `BASEPATH`, `pickle_load`。

**実行環境**: **safe**(t5chem には matplotlib/seaborn が無く不可。safe は全依存が揃い `func.figure_func` も import 可)。evaluate.sh と同じ。

**制約**: `src/recursion_train/` 以外は変更しない。例外は Step 1(build_dataset.py への `target_frags.pkl` 出力の1行追加。性能図の dup_frags 軸算出に必要)。

## dup_frags 性能図と target_frags

dup_frags の制約軸(重複数)は `dup_frags_analyze(target_frags)` で算出し、そのラウンドで dup_frags を再生成した際の `target_frags` 集合が必要。build_dataset.py は既に `target_frag_set` を計算済み(現行 146-151 行付近)なので、それを `pickle_save` で保存するだけ(**生成データの内容は不変。ファイルが1つ増えるのみ**)。

> 注意: `src/recursion_train/build_dataset.py` には現在ユーザーのローカル未コミット編集がある。Step 1 の実装はその作業ツリー版に1行追記する形になり、コミット時にユーザー編集も同一コミットに含まれる。分離したい場合は **事前にユーザーが build_dataset.{py,sh} をコミット**しておくこと。

## 出力パス

- 性能図: `figures/constraints/rffmg/t5chem/finetuning/{frag}/recursion/recursion{N}/beam/{scenario}/{metric}.png`
- 学習曲線: `figures/learning_curves/rffmg/t5chem/finetuning/{frag}/recursion/recursion{N}/curve.png`
- 訓練分布: `figures/constraints/train/rffmg/{frag}/recursion/recursion{N}/{scenario}/train.png`

## Plan

### Step 1: build_dataset.py に target_frags.pkl 出力を追加

- **Target file**: `src/recursion_train/build_dataset.py`(既存を編集)
- **Changes**: `func.utility` の import に `pickle_save` を追加。dup_frags 再生成の出力箇所(`{out_data_dir}/dup_frags/` を作る所)で `pickle_save(f'{out_data_dir}/dup_frags/target_frags.pkl', target_frag_set)` を1行追加(`make_datasets.py:264` と同じ)。**他の出力・ロジックは不変**。
- **Dependencies**: none

### Step 2: figure.py(recursion 図生成)

- **Target file**: `src/recursion_train/figure.py`(新規)
- **Changes**:
  - argparse: `--frag_method {brics,rc_cms}`(required)、`--recursion_num`(int, required)。scenarios=['frag_num','dup_frags','attach_point_num'] 固定。
  - import(`sys.path` に `src/` 追加): `from figure import read_wandb_loss_history, plot_learning_curve, attach_points_analyze, dup_frags_analyze, dup_frags_analyze_train, frag_num_analyze`、`from func.figure_func import create_boxplot, plot_single_dataset_pdf`、`from func.utility import BASEPATH, pickle_load`。
  - **(1) 性能図**(figure.py 325-366 を複製):
    - `result_dir = {BASEPATH}/results/rffmg/t5chem/finetuning/{frag}/recursion/recursion{N}/beam`。
    - 各 const_name について analyze_func / col_name / hue を設定(attach_point_num→`attach_points_analyze`/`max_attach_point_num`/hue=`add_frags_num`、dup_frags→`dup_frags_analyze(pickle_load(recursion{N}/dup_frags/target_frags.pkl))`/`dup_frags`/hue=`add_frags_num`、frag_num→`frag_num_analyze`/`frag_num`/hue=None)。
    - `{result_dir}/{const_name}/curated_data.tsv` を読み、`curated_df[[col_name,'add_frags_num']] = curated_df['fragment'].apply(lambda x: pd.Series(analyze_func(x)))`、metrics 4種を `create_boxplot` で `figures/constraints/rffmg/t5chem/finetuning/{frag}/recursion/recursion{N}/beam/{const_name}/{metric}.png` に出力。
  - **(2) 学習曲線**(figure.py 428-434 を複製、当該ラウンドに限定):
    - `glob({BASEPATH}/wandb/rffmg/t5chem/finetuning/{frag}/recursion/recursion{N}/**/latest-run/run-*.wandb', recursive=True)` を対象に `read_wandb_loss_history` → `plot_learning_curve` で `figures/learning_curves/rffmg/t5chem/finetuning/{frag}/recursion/recursion{N}/curve.png` に出力。
  - **(3) 訓練データの分布**(figure.py 368-405 を複製):
    - `train.source = {BASEPATH}/data/rffmg/{frag}/recursion/recursion{N}/normal/train.source` を読む。
    - 各 const_name について analyze_func(attach_point_num→`attach_points_analyze`、dup_frags→`dup_frags_analyze_train`(target 不要)、frag_num→`frag_num_analyze`)で col を算出、`plot_single_dataset_pdf`(density=False, y='Number of compounds')で `figures/constraints/train/rffmg/{frag}/recursion/recursion{N}/{const_name}/train.png` に出力。
  - 入力ファイルが無い場合は `os.path.exists` で skip(figure.py と同様)。
- **Dependencies**: after Step 1(dup_frags 性能図が target_frags.pkl を要するため。学習曲線・訓練分布は Step 1 非依存)

### Step 3: figure.sh(safe 環境ラッパー)

- **Target file**: `src/recursion_train/figure.sh`(新規)
- **Changes**: リポジトリルートへ cd、**safe 環境**を使う(`~/miniconda3/envs/safe/bin/python`、evaluate.sh と同方式)。先頭変数 `FRAG_METHOD="brics"` / `RECURSION_NUM=1`。`figure.py` を各引数で実行。
- **Dependencies**: after Step 2

## Notes / 前提

- 図生成の前に、当該ラウンドの evaluate(`curated_data.tsv`)と build_dataset(train.source / target_frags.pkl)、train(wandb ログ)が揃っていること。
- コードスタイル(CLAUDE.md)遵守。実装は承認後 `implementer` に委譲。main へ直接コミット。
- build_dataset.py のユーザー未コミット編集の扱いは上記「dup_frags 性能図と target_frags」注意を参照。
