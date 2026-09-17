# Representation for Flexible Fragment-Controlled Molecular Generation (RFFMG): A Framework for Versatile Substructure-Conditioned Molecular Design

[Japanese version (日本語版)](README_ja.md)

```bash
git clone https://github.com/sa-akinori/rffmg_molecular_design.git
cd rffmg_molecular_design
```

## Tutorial

Separate tutorials provide step-by-step instructions for extracting fragments from arbitrary SMILES and generating molecules using pre-trained models:

- [RFFMG tutorial](tutorial/tutorial_rffmg.ipynb) — use the `rffmg` kernel.
- [SAFE tutorial](tutorial/tutorial_safe.ipynb) — use the `safe` kernel.
- [PromptSMILES tutorial](tutorial/tutorial_promptsmiles.ipynb) — use the `promptsmiles` kernel.

## Three Conda Environments Required

One environment per method. `pip install -e .` installs the local `func` package and is required in
**every** environment.

### RFFMG (T5Chem/GPT2 and dataset construction)
```bash
conda create -n rffmg python=3.12.12
conda activate rffmg
pip install -r requirements/t5chem_requirements.txt
pip install -e .
```
### SAFE (also used by evaluation)
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
| Method | Representation | Base model | Environment |
|---|---|---|---|
| RFFMG (T5Chem) | `fragments >> molecule` | T5 (~14.8M) | `rffmg` |
| RFFMG (GPT2) | `fragments >> molecule` | `entropy/gpt2_zinc_87m` (~87M) | `rffmg` |
| SAFE | SAFE string | safe-gpt (~88.8M) | `safe` |
| PromptSMILES | plain SMILES + inference-time prompting | `entropy/gpt2_zinc_87m` | `promptsmiles` |

Every `run_*.sh` and `gen_*.sh` script activates its own environment, so they can be launched from
any shell. The `python src/...` commands below have to be run in the environment shown next to them.

## Modifications to Virtual Environments
## T5Chem
### Speed up training (t5chem/run_trainer.py)
```python
# compute_metrics = AccuracyMetrics
compute_metrics = None
```

### Clarify model save paths (t5chem/run_trainer.py)
```python
# tokenizer.save_vocabulary(args.output_dir)
# trainer.save_model(args.output_dir)
os.makedirs(f'{args.output_dir}/best_model/')
tokenizer.save_vocabulary(f'{args.output_dir}/best_model/')
trainer.save_model(f'{args.output_dir}/best_model/')
```

## SAFE
### Clarify model save paths (safe/trainer/cli.py)
```python
# trainer.save_model()
trainer.save_model(os.path.join(training_args.output_dir, "best_model"))

# tokenizer.save(os.path.join(training_args.output_dir, "tokenizer.json"))
tokenizer.save(os.path.join(training_args.output_dir, "best_model/tokenizer.json"))
```
### Add early stopping for faster training (safe/trainer/cli.py)
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
### Fix errors caused by transformers version (safe/trainer/trainer_utils.py & safe/tokenizer.py)
```python
# Error in safe/trainer/trainer_utils.py (line 19)
# def compute_loss(self, model, inputs, return_outputs=False):
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

# Error in safe/tokenizer.py (line 290)
# self.tokenizer.save_pretrained(*args, **kwargs)
self.tokenizer.save(*args, **kwargs)
```

## Preparing Pre-trained/Trained Models and Datasets
### Download trained models from Hugging Face
```bash
$ python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sato-akinori/FFMG', allow_patterns='models/*', local_dir='.')"
$ find models -name "*.zip" -exec sh -c 'unzip -o "$1" -d "$(dirname "$1")" && rm "$1"' _ {} \;
```

### T5Chem
Download and extract the pre-trained model.
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

### Curated Dataset
Coming soon.

## Building Datasets
### First Step
```bash
$ conda activate rffmg
$ python src/curate_datasets.py
```

### Creating Datasets

In the current `src/make_datasets.py`, the train/validation/test split construction and
SAFE/PromptSMILES dataset saving code are commented out. The active sections build the
`dup_frags` and `attach_point_num` evaluation datasets and require existing
`unique_frags.csv` and `normal/train.source` files. The commands below alone do not
create the complete set of datasets needed for training.

```bash
# 1. Create RFFMG fragments
$ conda activate rffmg
$ python src/gen_frags/rffmg_frags.py --frag_method brics # choose brics or rc_cms

# 2. Create SAFE fragments
$ conda activate safe
$ python src/gen_frags/safe_frags.py --frag_method brics # choose brics or rc_cms

# 3. Build dup_frags and attach_point_num evaluation datasets from existing data
$ conda activate safe
$ python src/make_datasets.py --frag_method brics # choose brics or rc_cms
```

The datasets used for training and generation are stored at the following paths.
For RFFMG, `{N}times_sampling` corresponds to `--sampling_num` (default: 5).

| Dataset | Path | Content |
|---|---|---|
| RFFMG | `data/rffmg/{frag}/{N}times_sampling/normal/` | `train/val/test.source` + `.target` |
| SAFE | `data/safe/{frag}/normal` | HF `DatasetDict` (`smiles`, `full_safe`, `pass_safe`, `full_fragments`, `pass_fragments`) |
| PromptSMILES | `data/promptsmiles/{frag}/normal` | HF `DatasetDict` (`smiles`, `pass_fragments`) |

Use the same molecule split and fragment sets when comparing methods. Before comparing
results row by row, check the row counts and the correspondence between targets and fragment sets.

### Model Training
```bash
# 1. Train the RFFMG model (T5Chem or GPT2)
$ bash src/train_model/run_rffmg.sh
# Set MODEL_NAME/MODE/FRAG_NAME inside the .sh.
#   MODEL_NAME="t5chem": T5Chem (runs `t5chem train`).
#   MODEL_NAME="gpt":    GPT2 (src/train_model/train_gpt.py). MODE="finetuning" starts from
#                        entropy/gpt2_zinc_87m; MODE="from_scratch" uses the same config with random weights.

# 2. Train SAFE-GPT
# Set FRAG_NAME/MODE/PRETRAINED_DIR at the top of the .sh before running it.
# MODE="finetuning": start from the pre-trained weights in PRETRAINED_DIR.
# MODE="from_scratch": use the config and tokenizer in that directory with randomly initialized weights.
$ bash src/train_model/run_safe.sh

# 3. Train the PromptSMILES prior (plain-SMILES language model)
$ bash src/train_model/run_promptsmiles.sh
# Set FRAG_NAME/MODE at the top of the .sh.
# The prior is an unconditional SMILES language model; PromptSMILES supplies its prompt only at
# inference time. Each molecule is always rewritten from a random root atom, because the prompts
# seen at inference are non-canonical and start at an arbitrary atom. No augmentation is applied
# (one molecule = one sequence), and the randomization is drawn once when the dataset is built.
```

For RFFMG-GPT and PromptSMILES, `MODE="finetuning"` starts from `entropy/gpt2_zinc_87m`,
and `MODE="from_scratch"` uses the same config with random weights.
The three GPT2-based methods share the same hyperparameters (LR 1e-4, 50 epochs,
batch 32, warmup 10000, eval/save every 5000 steps, early stopping patience 15, seed 42), so the
comparison isolates the representation rather than the training budget.

### Molecular Generation

All methods are prompted with the **same fragment sets** (the `pass_fragments` of the shared
test split, attachment points written as bare `*` with no connectivity information) and write
`predictions.csv` with the columns `target`, `prediction_1` .. `prediction_N`, so the shared
evaluation pipeline reads them unchanged.

```bash
# RFFMG model
$ bash src/gen_mols/gen_rffmg.sh

# SAFE-GPT model
$ bash src/gen_mols/gen_safe.sh

# PromptSMILES model
$ bash src/gen_mols/gen_promptsmiles.sh
```

Results are written to `results/{repr}/{model}/{model_ver}/{frag}/{gen_method}/{additional_path}/`.

### Evaluation of Generated Molecules

Currently, `src/evaluation.py` disables the basic evaluation with `if 0` and enables
only the JS divergence comparison with `if 1`. The command below therefore does not
produce `stats.csv` with the current settings. Before running the basic evaluation,
change the `if 0` immediately above `# Load dataset` to `if 1`, and change the
`if 1: # Calculate js-divergence between train and test` block to `if 0`.

```bash
$ conda activate safe
$ python src/evaluation.py --repr_name rffmg --model_name gpt --model_ver finetuning --frag_method rc_cms --additional_path normal
```

`--repr_name` / `--model_name` combinations covered in this README:

| `--repr_name` | `--model_name` | results path |
|---------------|----------------|--------------|
| rffmg         | t5chem         | `results/rffmg/t5chem/` |
| rffmg         | gpt            | `results/rffmg/gpt/` |
| safe          | gpt            | `results/safe/gpt/` |
| promptsmiles  | gpt            | `results/promptsmiles/gpt/` |

The basic evaluation requires generated `predictions.csv` files and training data.
For RFFMG, the corresponding `test.source` is joined by row number, so its row count and order
must match the generation results. SAFE and PromptSMILES use the `fragment` column in
`predictions.csv` and do not read `test.source`.

To run the JS divergence comparison, prepare the physicochemical-property CSVs
for RFFMG (fine-tuned T5Chem and GPT) and SAFE (pretrained and fine-tuned GPT), plus
the training molecules. This block uses that fixed comparison regardless of
`--repr_name`, `--model_name`, or `--model_ver`; `--frag_method` and `--sampling_num`
select the corresponding input paths.

The training-property CSV paths currently differ between the basic evaluation and
the JS comparison:

| Operation | Path |
|---|---|
| Basic evaluation: existence check | `results/physical_properties/train/{N}times_sampling/physical_property.csv` |
| Basic evaluation: save (RFFMG only) | `results/physical_properties/train/{N}times_sampling/train_physical_property.csv` |
| JS comparison: read | `results/train_physical_property.csv` |

Before enabling the JS comparison, align these references to the CSV computed from
the training data being compared. Enabling the basic evaluation alone does not
resolve this path mismatch.

When enabled, the basic evaluation writes `stats.csv` using every row of the input
`predictions.csv`. PromptSMILES generates nothing for fragment sets it cannot express;
those rows contain `INVALID_SMILES` and receive a score of zero, as with SAFE decoding
failures. These metrics therefore include coverage.

Which rows were left ungenerated is recorded in the `sampler` column of `predictions.csv`
(`scaffold` / `linking` / `unsupported` / `invalid_target` / `generation_error`); the counts per
reason are written to `generation_params.txt`.

If you need the curated ChEMBL dataset used in this study, please feel free to contact us at [sato.akinori@naist.ac.jp] or [miyao@dsc.naist.jp].
