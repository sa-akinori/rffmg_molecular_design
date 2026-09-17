# Representation for Flexible Fragment-Controlled Molecular Generation (RFFMG): A Framework for Versatile Substructure-Conditioned Molecular Design
```bash
git clone https://github.com/sa-akinori/rffmg_molecular_design.git
cd rffmg_molecular_design
```

## チュートリアル

分子生成のチュートリアルは手法ごとに分かれています。

- [RFFMGチュートリアル](tutorial/tutorial_rffmg.ipynb)：`t5chem` カーネルを使用します。
- [SAFEチュートリアル](tutorial/tutorial_safe.ipynb)：`safe` カーネルを使用します。
- [PromptSMILESチュートリアル](tutorial/tutorial_promptsmiles.ipynb)：`promptsmiles` カーネルを使用します。

学習済みモデルを用いて、任意のSMILESからフラグメントを抽出し、新しい分子を生成する手順をステップごとに解説しています。

## 3つの仮想環境が必要

手法ごとに1つの環境を用意します。`pip install -e .`（ローカルの `func` パッケージ）は
**すべての環境で必要**です。

### T5Chem（RFFMG-GPT の学習とデータセット構築にも使用）
```bash
conda create -n t5chem python=3.12.12
conda activate t5chem
pip install -r requirements/t5chem_requirements.txt
pip install -e .
```
### SAFE（生成分子の評価にも使用）
```bash
conda create -n safe python=3.12.12
conda activate safe
pip install -r requirements/safe_requirements.txt
pip install -e .
```
### PromptSMILES
```bash
conda create -n promptsmiles python=3.12.12
conda activate promptsmiles
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/promptsmiles_requirements.txt
pip install -e .
```

| 手法 | 表現 | ベースモデル | 環境 |
|---|---|---|---|
| RFFMG (T5Chem) | `断片集合 >> 分子` | T5 (~14.8M) | `t5chem` |
| RFFMG (GPT2) | `断片集合 >> 分子` | `entropy/gpt2_zinc_87m` (~87M) | `t5chem` |
| SAFE | SAFE 文字列 | safe-gpt (~88.8M) | `safe` |
| PromptSMILES | プレーンな SMILES + 推論時プロンプト | `entropy/gpt2_zinc_87m` | `promptsmiles` |

`run_*.sh` と `gen_*.sh` はすべてスクリプト内で環境を activate するため、どのシェルからでも実行できます。
以下の `python src/...` のコマンドは、併記した環境で実行してください。

## 仮想環境の変更点
## T5Chem
### 学習速度向上のための変更(t5chem/run_trainer.py)
```python
# compute_metrics = AccuracyMetrics
compute_metrics = None
```

### モデルの保存をわかりやすくするための変更(t5chem/run_trainer.py)
```python
# tokenizer.save_vocabulary(args.output_dir)
# trainer.save_model(args.output_dir)
os.makedirs(f'{args.output_dir}/best_model/')
tokenizer.save_vocabulary(f'{args.output_dir}/best_model/')
trainer.save_model(f'{args.output_dir}/best_model/')
```

## SAFE
### モデルの保存をわかりやすくするための変更(safe/trainer/cli.py)
```python
# trainer.save_model()
trainer.save_model(os.path.join(training_args.output_dir, "best_model"))

# tokenizer.save(os.path.join(training_args.output_dir, "tokenizer.json"))
tokenizer.save(os.path.join(training_args.output_dir, "best_model/tokenizer.json"))
```
### 学習高速化のための追加(safe/trainer/cli.py)
```python
trainer = SAFETrainer(
    model=model,
    tokenizer=None,  # we don't deal with the tokenizer at all, https://github.com/huggingface/tokenizers/issues/581 -_-
    train_dataset=train_dataset.shuffle(seed=(training_args.seed or 42)),
    eval_dataset=dataset.get(eval_dataset_key_name, None),
    args=training_args,
    prop_loss_coeff=model_args.prop_loss_coeff,
    compute_metrics=compute_metrics if training_args.do_eval else None,
    data_collator=data_collator,
    preprocess_logits_for_metrics=(
        preprocess_logits_for_metrics if training_args.do_eval else None
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=15)] #add
)
```
### transformersのバージョンによってエラーが出るので修正してください。(safe/trainer/trainer_utils.py & safe/tokenizer.py)
```python
# safe/trainer/trainer_utils.py(19行目)におけるエラー
# def compute_loss(self, model, inputs, return_outputs=False):
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

# safe/tokenizer.py(290行目)におけるエラー
# self.tokenizer.save_pretrained(*args, **kwargs)
self.tokenizer.save(*args, **kwargs)
```

## Pre-trained/trainedモデル・データセットの準備
### 本研究の学習済みモデルをHugging Faceからmodelsフォルダーをダウンロード
```bash
$ python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sato-akinori/FFMG', allow_patterns='models/*', local_dir='.')"
$ find models -name "*.zip" -exec sh -c 'unzip -o "$1" -d "$(dirname "$1")" && rm "$1"' _ {} \;
```

### T5Chem
事前学習モデルをダウンロードして解凍する。
```bash
$ mkdir -p models/rffmg/t5chem/pretrained
$ wget -P models/rffmg/t5chem/pretrained https://zenodo.org/records/14280768/files/simple_pretrain.tar.bz2
$ tar -xjvf models/rffmg/t5chem/pretrained/simple_pretrain.tar.bz2 --strip-components=3 -C models/rffmg/t5chem/pretrained/
```

### SAFE
```bash
$ mkdir -p models/safe/gpt/pretrained
$ git clone https://huggingface.co/datamol-io/safe-gpt/ models/safe/gpt/pretrained/
```

### curated datasetの準備
準備中

## データセットの構築
### 最初のステップ
```bash
$ conda activate t5chem
$ python src/curate_datasets.py
```

### データセットの作成

現在の `src/make_datasets.py` では、訓練・検証・テスト分割の作成と
SAFE・PromptSMILESデータセットの保存処理がコメントアウトされています。
有効なのは `dup_frags` と `attach_point_num` の評価用データ作成で、
既存の `unique_frags.csv` と `normal/train.source` が必要です。
以下のコマンドだけでは、学習に必要なデータセット一式は作成されません。

```bash
# 1. rffmgフラグメントの作成
$ conda activate t5chem
$ python src/gen_frags/rffmg_frags.py --frag_method brics # chose brics or rc_cms

# 2. safeフラグメントの作成
$ conda activate safe
$ python src/gen_frags/safe_frags.py --frag_method brics # chose brics or rc_cms

# 3. 既存データからdup_frags・attach_point_numの評価用データを作成
$ conda activate safe
$ python src/make_datasets.py --frag_method brics # chose brics or rc_cms
```

学習・生成で使用するデータセットの配置は次のとおりです。
RFFMGの `{N}times_sampling` は `--sampling_num`（既定値5）に対応します。

| データセット | パス | 内容 |
|---|---|---|
| RFFMG | `data/rffmg/{frag}/{N}times_sampling/normal/` | `train/val/test.source` + `.target` |
| SAFE | `data/safe/{frag}/normal` | HF `DatasetDict`（`smiles`, `full_safe`, `pass_safe`, `full_fragments`, `pass_fragments`） |
| PromptSMILES | `data/promptsmiles/{frag}/normal` | HF `DatasetDict`（`smiles`, `pass_fragments`） |

手法間の比較には、同じ分子分割・断片集合を使用してください。
行単位で比較する場合は、使用するデータの行数と `target`・断片集合の対応を確認してください。

### モデルの学習
```bash
# 1. rffmgモデルの学習（T5ChemまたはGPT2）
$ bash src/train_model/run_rffmg.sh
# .shファイル内の MODEL_NAME/MODE/FRAG_NAME を設定してください。
#   MODEL_NAME="t5chem": T5Chem（`t5chem train` を実行）。
#   MODEL_NAME="gpt":    GPT2（src/train_model/train_gpt.py）。MODE="finetuning" は entropy/gpt2_zinc_87m を初期値に、
#                        MODE="from_scratch" は同一configをランダム初期化で学習します。

# 2. SAFE-GPTの学習
# .sh上部の FRAG_NAME/MODE/PRETRAINED_DIR を設定してから実行してください。
# MODE="finetuning": PRETRAINED_DIRの事前学習済み重みから学習します。
# MODE="from_scratch": 同ディレクトリのconfig・tokenizerを使用し、重みはランダム初期化します。
$ bash src/train_model/run_safe.sh

# 3. PromptSMILES の prior（プレーンSMILESの言語モデル）の学習
$ bash src/train_model/run_promptsmiles.sh
# FRAG_NAME/MODE は .sh 上部で設定してください。
# prior は無条件のSMILES言語モデルで、PromptSMILES は推論時にのみプロンプトを与えます。
# 推論時のプロンプトは非カノニカルで任意の原子から始まるため、各分子は常にランダムな根原子から
# 書き直します。データ水増しは行わず（1分子=1系列）、ランダム化はデータセット構築時に1回だけ行います。
```

RFFMG-GPTとPromptSMILESでは、`MODE="finetuning"` は `entropy/gpt2_zinc_87m` を初期値に、
`MODE="from_scratch"` は同一configをランダム初期化して学習します。GPT2ベースの3手法はハイパーパラメータを統一しており
（LR 1e-4 / 50エポック / batch 32 / warmup 10000 / eval・save 5000ステップごと /
EarlyStopping patience 15 / seed 42）、学習量ではなく表現の違いを比較できるようにしています。

### 分子の生成

各手法とも**同一の断片集合**（共有 test split の `pass_fragments`。結合点は素の `*` で、
どの断片同士がつながるかの情報は含まない）をプロンプトとし、
`target`, `prediction_1` .. `prediction_N` の列を持つ `predictions.csv` を出力します。
共通の評価パイプラインがそのまま読めます。

```bash
# rffmgモデル
$ bash src/gen_mols/gen_rffmg.sh

# safe-gptモデル
$ bash src/gen_mols/gen_safe.sh

# PromptSMILESモデル
$ bash src/gen_mols/gen_promptsmiles.sh
```

生成結果は `results/{表現}/{モデル}/{model_ver}/{frag}/{gen_method}/{additional_path}/` に出力されます。

### 生成分子の評価

現在の `src/evaluation.py` は、基本集計が `if 0` で無効、JS divergence の比較のみが
`if 1` で有効になっています。そのため、現在の設定で下記コマンドを実行しても
`stats.csv` は作成されません。基本集計を実行する前に、`# Load dataset` の直前にある
`if 0` を `if 1` にし、`if 1: # Calculate js-divergence between train and test` の
ブロックを `if 0` に変更してください。

```bash
$ conda activate safe
$ python src/evaluation.py --repr_name rffmg --model_name gpt --model_ver finetuning --frag_method rc_cms --additional_path normal

```

本READMEで扱う `--repr_name` / `--model_name` の組み合わせ:

| `--repr_name` | `--model_name` | 結果のパス |
|---------------|----------------|-----------|
| rffmg         | t5chem         | `results/rffmg/t5chem/` |
| rffmg         | gpt            | `results/rffmg/gpt/` |
| safe          | gpt            | `results/safe/gpt/` |
| promptsmiles  | gpt            | `results/promptsmiles/gpt/` |

基本集計には生成済みの `predictions.csv` と訓練データが必要です。
RFFMGでは対応する `test.source` を行番号で結合するため、生成結果と行数・行順を揃えてください。
SAFE・PromptSMILESでは `predictions.csv` 内の `fragment` 列を使い、`test.source` は読みません。

JS divergence の比較を実行する場合は、RFFMG（T5Chem・GPTの学習済みモデル）、
SAFE（GPTの事前学習モデル・追加学習モデル）、訓練分子の物性CSV一式が必要です。
この比較対象は `--repr_name`、`--model_name`、`--model_ver` によらず固定されており、
`--frag_method` と `--sampling_num` が読み込むファイルのパスに反映されます。

また、訓練分子の物性CSVは、基本集計とJS比較で次のようにパスが異なっています。

| 処理 | パス |
|---|---|
| 基本集計の存在確認 | `results/physical_properties/train/{N}times_sampling/physical_property.csv` |
| 基本集計の保存先（RFFMGのみ） | `results/physical_properties/train/{N}times_sampling/train_physical_property.csv` |
| JS比較の読み込み先 | `results/train_physical_property.csv` |

JS比較を有効にする前に、比較対象の訓練データから計算したCSVへ各参照を揃えてください。
基本集計を有効にするだけでは、このパスの不一致は解消しません。

基本集計を有効にすると、入力した `predictions.csv` の全行を対象に `stats.csv` を出力します。
PromptSMILES は表現できない断片集合に対しては生成を行わず、その行は `INVALID_SMILES` として0点で
集計されます（SAFE のデコード失敗と同じ扱いです）。したがってこの数値は
カバレッジを含んだものになります。

どの行が生成に回らなかったかは `predictions.csv` の `sampler` 列に記録されます
（`scaffold` / `linking` / `unsupported` / `invalid_target` / `generation_error`）。
理由別の件数は `generation_params.txt` にあります。

If you need the curated ChEMBL dataset used in this study, please feel free to contact us at [sato.akinori@naist.ac.jp] or [miyao@dsc.naist.jp].
