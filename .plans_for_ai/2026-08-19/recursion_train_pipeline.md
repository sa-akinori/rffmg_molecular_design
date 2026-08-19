# Plan: recursion_train パイプライン(データ作成 / 訓練 / 生成 / 評価)

- **Date**: 2026-08-19
- **Status**: pending-approval (rev.3 — リーク回避: 成功ペアは train のみに追加、val/test から重複分子を除去、3シナリオ test を再生成)

## Overview

ロバスト性シナリオ(`frag_num` / `dup_frags` / `attach_point_num`)で **生成に成功した分子**(`curated_data.tsv` の `valid_smis_on_frags`)を訓練データに累積追加し、rffmg × t5chem(finetuning)を再学習→再生成→再評価する「再帰的自己学習」パイプラインを `src/recursion_train/` に構築する。

**リーク回避の核心**(rev.2 からの是正):
- 旧設計は「シナリオ test のフラグメント集合 F → 分子 M」ペアを train に追加しつつ、同じ F で test していた(= train と test が重複)。
- rev.3 では **(B) 方式**を採る:
  1. 成功ペア(F→M)を **train のみ**に累積追加。
  2. 3シナリオ(frag_num/dup_frags/attach_point_num)の test を、**make_datasets.py の該当ロジックを複製して再生成**(拡張後 train.source と dedup、target 空)。→ train に入った F は自動除外され、新しい test は train と重複しない。
  3. `normal` は baseline 固定(再分割しない)。ただし **生成分子と同一の分子が val/test にあれば val/test から除去**(train へは移さない)。
  4. `frag_order` は作らない。SAFE/promptsmiles/fraggpt 用データも作らない(rffmg のみ)。

**スコープ / 制約**:
- 対象は rffmg × t5chem × finetuning のみ。`sampling='5times_sampling'` 固定、frag_method は `.sh` 先頭変数(既定 brics)。
- ラベル `recursion{N}`(N=1,2,…)を各スクリプトに渡す。
- `src/recursion_train/` 以外の `src/` は変更しない(subprocess/import で再利用)。
- 汎用関数(`func.utility.BASEPATH`/`load_file`/`save_file`/`canonical_smiles`、`func.fragmentation.GetNHA`、`func.evaluation_func.*`)を再利用。**make_datasets.py は import しない**(先頭で `datasets` を import しており t5chem 環境で失敗し得るため、シナリオ生成ロジックは複製する)。

## recursion{N} データレイアウト(build_dataset.py が生成)

```
data/rffmg/{frag}/recursion/recursion{N}/
├── normal/            train.{source,target}=前ラウンドtrain + 成功ペア(F→M)(累積),
│                      val/test.{source,target}=前ラウンドval/test から「分子が成功集合に含まれる行」を除去
├── frag_num/          test.{source,target}=make_datasets複製で再生成(target空)
├── dup_frags/         test.{source,target}=同上
└── attach_point_num/  test.{source,target}=同上
```

- 累積の起点: N==1 → baseline `data/rffmg/{frag}/5times_sampling/`、N≥2 → `.../recursion/recursion{N-1}/`。
- 収穫元 `curated_data.tsv`: N==1 → `results/rffmg/t5chem/finetuning/{frag}/5times_sampling/beam/{scenario}/`、N≥2 → `.../recursion/recursion{N-1}/beam/{scenario}/`。
- シナリオ再生成で使う `unique_frags.csv`(全体フラグメントプール)は分子が増えないため baseline の `data/rffmg/{frag}/5times_sampling/unique_frags.csv` を流用。

## 出力パス(model / results)

- model: `models/rffmg/t5chem/finetuning/{frag}/recursion/recursion{N}/best_model`
- results: `results/rffmg/t5chem/finetuning/{frag}/recursion/recursion{N}/beam/{scenario}/`

## 運用フロー(手動、ドライバなし)

`(baseline 評価済み)` → build_dataset r1 → train r1 → generate r1 ×4条件 → evaluate r1 ×4条件 → build_dataset r2(r1 の curated を収穫)→ …

## Plan

### Step 1: build_and_train.{py,sh} を削除

- **Target file**: `src/recursion_train/build_and_train.py`, `src/recursion_train/build_and_train.sh`
- **Changes**: `git rm` で削除(build_dataset + train に置換)。
- **Dependencies**: none

### Step 2: build_dataset.py(データ作成専用)

- **Target file**: `src/recursion_train/build_dataset.py`(新規)
- **Changes**:
  - argparse: `--frag_method {brics,rc_cms}`(required)、`--recursion_num`(int, required)、`--n_select`(int, 既定5)、`--seed`(int, 既定0)。scenarios / sampling 固定。
  - `func.utility` から `BASEPATH`/`load_file`/`save_file`/`canonical_smiles`、`func.fragmentation` から `GetNHA` を import(`sys.path` に `src/` 追加)。make_datasets の module 直下の軽量ヘルパ(`unique_f_num`, `countAtttachPoint`)は複製。
  - **(a) 成功ペアの収穫**: 収穫元 results dir(N==1/N≥2 で分岐)の各 `{scenario}/curated_data.tsv` を `pd.read_csv(sep='\t', index_col=0)` で読み、`nvalid_onfrags==0` はスキップ、`ast.literal_eval(valid_smis_on_frags)` から **単一共有 rng**(`rng = random.Random(seed)`)で `rng.sample(smis, min(n_select, len(smis)))` を抽選、`(fragment, canonical(smi))` を蓄積。`added` DataFrame(`source,target,scenario`)を `(source,target)` dedup。`SUCCESS_MOLS = set(added['target'])`(canonical 化済み)。
  - **(b) normal の生成**(起点: N==1 は baseline normal、N≥2 は recursion{N-1}/normal):
    - added を起点 train の `(source,target)` 集合と dedup。
    - `recursion{N}/normal/train.{source,target}` = 起点 train + added(**累積**)。
    - `recursion{N}/normal/val.{source,target}` = 起点 val から、`canonical(target) in SUCCESS_MOLS` の行を除去。
    - `recursion{N}/normal/test.{source,target}` = 起点 test から同様に除去。
  - **(c) 3シナリオ test の再生成**(make_datasets.py:175-305 を複製):
    - `unique_frags.csv`(baseline)を読み、拡張後 `recursion{N}/normal/train.source` を canonical 化して dedup 集合に。
    - `frag_num`(175-203)/`dup_frags`(205-265)/`attach_point_num`(267-305)の各サンプリング・seed・サイズ・train dedup を忠実に複製し、`recursion{N}/{scenario}/test.source` と **空 target** `test.target` を出力。
  - **(d) ログ**: `added_pairs.csv` と `augmentation_log.txt`(picked件数・dedup件数・除去した val/test 行数・train サイズ)。
- **Dependencies**: after Step 1

### Step 3: build_dataset.sh(t5chem 環境ラッパー)

- **Target file**: `src/recursion_train/build_dataset.sh`(新規)
- **Changes**: `SCRIPT_DIR=...`、リポジトリルートへ cd、`conda activate t5chem`。先頭変数 `FRAG_METHOD="brics"` / `RECURSION_NUM=1` / `N_SELECT=5` / `SEED=0`。`build_dataset.py` を実行。
- **Dependencies**: after Step 2

### Step 4: train.py(訓練専用)

- **Target file**: `src/recursion_train/train.py`(新規)
- **Changes**: argparse(`--frag_method`/`--recursion_num`)。`BASEPATH` を import。`subprocess.run(check=True)` で `t5chem train --pretrain {BASEPATH}/models/rffmg/t5chem/pretrained --data_dir {BASEPATH}/data/rffmg/{frag}/recursion/recursion{N}/normal --output_dir {BASEPATH}/models/rffmg/t5chem/finetuning/{frag}/recursion/recursion{N} --task_type product --num_epoch 50`(`run_rffmg.sh:29-35` finetuning 分岐と同一。data_dir が `recursion{N}/normal`)。
- **Dependencies**: after Step 3(データが必要)

### Step 5: train.sh(t5chem 環境ラッパー)

- **Target file**: `src/recursion_train/train.sh`(新規)
- **Changes**: ルートへ cd、`conda activate t5chem`、先頭変数 `FRAG_METHOD`/`RECURSION_NUM`。`train.py` を実行。
- **Dependencies**: after Step 4

### Step 6: generate.py を更新(読み取り先を recursion ツリーへ)

- **Target file**: `src/recursion_train/generate.py`(既存を編集)
- **Changes**: `data_dir` を `.../recursion/recursion{N}/{additional_path}` に変更(旧: `5times_sampling/{additional_path}`)。model_dir / output_dir / t5chem predict 呼び出しは不変。
- **Dependencies**: after Step 5(モデルが必要)

### Step 7: evaluate.py を更新(読み取り先を recursion ツリーへ)

- **Target file**: `src/recursion_train/evaluate.py`(既存を編集)
- **Changes**: test.source を `.../recursion/recursion{N}/{additional_path}/test.source` に、novelty 基準 train を `.../recursion/recursion{N}/normal/train.target` に変更。outfd / グルー / `sc3_check_genmol_results` 呼び出しは不変。
- **Dependencies**: after Step 6(predictions が必要)

### 変更なし

- `generate.sh` / `evaluate.sh`(先頭変数・実行内容が同一)。

## Notes / 前提

- recursion1 の前に normal baseline(t5chem finetuning)の3シナリオ評価(`curated_data.tsv`)が済んでいること(本計画の対象外)。
- リーク基準は **分子レベル**: train↔val/test は分子で重複しない(val/test から成功分子を除去)。3シナリオ test は train.source と dedup 済み。
- コードスタイル(CLAUDE.md)遵守。実装は承認後 `implementer` に委譲。main へ直接コミット(worktree・作業ブランチは作らない)。
