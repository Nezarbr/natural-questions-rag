# RAG Question-Answering on Natural Questions

A retrieval pipeline built from scratch on a self-made Wikipedia corpus: stream Natural
Questions, clean the raw token streams into prose, chunk and embed, measure passage recall,
then fine-tune a small cross-encoder reranker on hard negatives mined by the retriever.

| | |
|---|---|
| corpus | 18,997 pages / 16,229 questions / 469,387 chunks |
| cleaning | 9 rules, **50.7%** of raw prose words removed, no answer span ever cut |
| document recall | R@1 64.3% · R@10 96.1% · R@20 98.4% (chunk-max, `bge-base-en-v1.5`) |
| passage recall | R@1 39.0% · R@100 96.3% — **finding the paragraph is the hard part** |
| reranker | `ms-marco-MiniLM-L-6-v2` (22M), fine-tuned: val R@5 **69.1% → 81.1%**, R@10 **79.8% → 88.6%** |
| held-out split | 1,626 questions, stratified by question type, grouped by page |

Every figure above is printed by the notebook that produced it. This repo holds the code
only — run the notebooks to regenerate the numbers, or read the metrics each one writes to
`results/run_<RUN_ID>/`.

## The notebooks, in order

| # | notebook | what it does | hardware |
|---|---|---|---|
| 1 | `1_stream.ipynb` | stream NQ, keep short answers (≤5 words), shard the raw pages | CPU |
| 2 | `2_clean.ipynb` | 9 cleaning rules → `clean_corpus.pkl` | CPU |
| 3 | `3_chunk_embed.ipynb` | chunk 200/50, embed all 469,387 chunks, mean vs max reduction | **GPU** |
| 4 | `4_chunk_recall.ipynb` | score every question against every chunk, save the top-100 pool | **GPU** |
| 5 | `5_split.ipynb` | train/val split, stratified and grouped | CPU |
| 6 | `6_finetune_rerank.ipynb` | fine-tune the cross-encoder on hard negatives | **GPU** |
| 7 | `7_rerank_val.ipynb` | all reranker candidates on the same val split | **GPU** |

Each notebook has the same shape: a markdown overview, one imports cell, one config cell,
function definitions (nothing executes), then orchestration. Each ends by pushing its
artifact to a Hugging Face dataset repo and writing a metrics JSON locally.

**Notebook 7 has not been run yet.** It is written and validated; it is the apples-to-apples
comparison of the fine-tuned MiniLM against `bge-reranker-v2-m3` on the same 1,626 val
questions, and until it runs any cross-model quality claim rests on a margin transferred
between samples of different difficulty.

## Running on a RunPod pod

Notebooks 1, 2 and 5 need no GPU. Notebook 2 is ~4 minutes of CPU for the whole corpus.
Notebooks 3, 4, 6 and 7 want a GPU — A40 or A6000 at ~$0.59/h have been the reliable
choices; A5000 at $0.27/h works when it is in stock.

### 1. Copy the code over

```bash
ssh root@<HOST> -p <PORT> 'mkdir -p /workspace/notebooks'
scp -P <PORT> notebooks/*.ipynb notebooks/nqrun.py root@<HOST>:/workspace/notebooks/
scp -P <PORT> requirements.txt root@<HOST>:/workspace/
```

`nqrun.py` must live in `/workspace/notebooks/`, not `/workspace/`. `nqrun.ROOT` is
`Path(nqrun.__file__).parent.parent`, so that placement puts artifacts in
`/workspace/artifacts` and run logs in `/workspace/results`. Put it one level up and `ROOT`
becomes `/`, scattering output across the filesystem root. `scp` uses `-P`; `ssh` uses `-p`.

### 2. Dependencies

The `runpod/pytorch` images ship an externally-managed Python (PEP 668), so a bare
`pip install` is refused. Use a venv that inherits the image's torch:

```bash
python -m venv --system-site-packages /workspace/venv
/workspace/venv/bin/pip install -q "huggingface-hub==0.26.2" "transformers==4.46.3" \
    "sentence-transformers==3.3.1" "accelerate==1.1.1" "hf-transfer==0.1.8"
```

`hf_transfer` is not optional. These images set `HF_HUB_ENABLE_HF_TRANSFER=1` in the
interactive login profile but ship without the package, so the first `hf_hub_download` dies
with `ValueError: Fast download using 'hf_transfer' is enabled ... but 'hf_transfer' package
is not available`. A probe run as `ssh host 'cmd'` does **not** see that variable and will
report it empty, which is how it gets missed.

`transformers` is pinned at 4.46.3 deliberately — it predates Qwen3, and the reader
experiments depend on that behaviour.

### 3. Credentials

```bash
export HF_HOME=/workspace/.cache/huggingface
/workspace/venv/bin/huggingface-cli login
```

The `HF_HOME` export must come first, or the token lands in `/root/.cache` where the run
cannot see it. The token needs **write** scope to push artifacts. It is per machine: a login
on a pod does not cover your laptop.

### 4. Run

A pod's `env` set at creation time does **not** reach an SSH session. `nqrun` reads
`NQ_RUN_ID` at *import* time, so if you forget the export it silently falls back to the date
and writes results into the wrong run directory:

```bash
cd /workspace/notebooks
export NQ_RUN_ID=<run-id>
setsid nohup /workspace/venv/bin/python -u <script>.py > /workspace/out.log 2>&1 < /dev/null &
```

`setsid nohup … < /dev/null &` detaches the run so it survives the SSH session closing — a
GPU job that dies with your laptop lid is an expensive mistake.

To get `<script>.py`, flatten the notebook:

```bash
python - <<'PY'
import json
nb = json.load(open('notebooks/4_chunk_recall.ipynb'))
cs = [(c.get('id'), ''.join(c['source'])) for c in nb['cells']
      if c['cell_type'] == 'code' and ''.join(c['source']).strip()]
open('4_chunk_recall.py','w').write(
    '\n\n'.join(f'# ===== {i} =====\n{s.rstrip()}' for i, s in cs) + '\n')
PY
```

Regenerate it after **every** notebook edit. A stale flattened script runs old code and
nothing warns you.

### 5. Stop the pod

A pod bills for every hour it is `RUNNING`, used or not, and its container disk is wiped on
restart — which is why notebook 1 uploads each shard as it lands rather than at the end.
Finish by confirming no pods, endpoints or volumes are left: a stopped pod still bills its
disk.

Two notes from experience: `stockStatus: "Low"` is a hint, not a reservation — a create can
still fail with "no longer any instances available", so have a fallback GPU ready. And
create the pod *after* you have decided what to run; one pod billed 72 minutes for 5 minutes
of compute because it idled during a discussion.

## Running locally

`nqrun` detects macOS and changes the defaults: `NQ_N_QUESTIONS=50`, `NQ_RUN_ID=local50`,
and **`NQ_HF_PUSH=0`, so uploads are silently skipped.** `NQ_HF_PUSH` is read at call time
and can be set any time before a push; `NQ_RUN_ID` is read at import time and must be set
before `import nqrun`.

Notebooks 2 and 5 run locally in minutes. Notebook 3 on CPU measures ~2 chunks/s, which is
about 83 hours for the full corpus — it has a throughput probe that tells you this in 20
seconds, before you commit.

## Artifacts

Each notebook pulls its inputs from a Hugging Face dataset repo and pushes its outputs back,
so no step depends on a previous pod's disk:

```
step1-checkpoint-v2/step1_checkpoint.pkl   1.4 GB   raw tokens, the lossless record
step1-shards/part_000..019.pkl             20 x 73 MB
step1-clean/clean_corpus.pkl               510 MB   the corpus
step1-clean/cleaning_audit.jsonl           per-page record of what each rule removed
step2-retrieval/chunk_emb.npy              1.44 GB  469,387 x 768
step2-retrieval/chunk_index.json           chunk -> page map and chunking parameters
step2-retrieval/top100_chunks.npy          the reranker's pool and its hard negatives
step4-rerank/split.json                    train/val indices
step4-rerank/minilm-ft/                     the fine-tuned reranker
```

Set the repo via `nqrun.HF_REPO`. **Chunk text is stored nowhere** — only embeddings. Every
notebook from 4 on rebuilds all 469,387 chunk strings with the same 200/50 chunker and then
asserts that each gold chunk contains its answer. That assert is the only guard against a
rebuild that drifts out of step with the saved embedding order, which would silently train
and evaluate on mismatched `(query, passage)` pairs.

## Limitations, stated plainly

- **Answers that sit inside tables are excluded.** NQ flattens tables into the token stream,
  and an audit of all 5,549 tables on one shard found a usable header row in only 26%, so
  any serialisation would mislabel three tables in four. 3,771 of 20,000 questions go this
  way. **Consequence: none of these numbers are comparable to published NQ-Open results**,
  nor to earlier versions of this project that included table answers.
- The short-answer cut-off of ≤5 words shapes which questions exist. Treat the dataset as
  "20,000 Natural Questions items filtered to short answers of ≤5 words", not as NQ-Open.
- Pages whose every question was excluded are **kept**, as distractors. Dropping them (19% of
  pages) would make retrieval easier than the benchmark it is modelled on.
- One gate in notebook 2 is vacuous: `spans lost entirely` prints a counter that is never
  incremented. The real count of fully-removed spans is 5,277, all table-borne. The two
  substantive gates — spans widened, and pages referenced but never seen — are real checks
  and both pass.
- Reader results are not included here. The best measured configuration reached 47% exact
  match and 64% LLM-judged correctness, but on 100 questions (SE ≈ ±5 points), which is too
  thin to publish.
