"""Shared run plumbing for the NQ-Open notebooks: paths, seeds, run log, metrics.

Every notebook imports this first. It replaces the Kaggle layout (/kaggle/working +
attached /kaggle/input datasets) with one local tree:

    <ROOT>/artifacts/<artifact-set>/...     big outputs (embeddings, models) - git-ignored
    <ROOT>/results/run_<RUN_ID>/...         run log + metrics JSON per notebook - committed

Artifact-set directory names keep the old Kaggle dataset slugs (step1-rag-artifacts-v2,
step4-ft-reranker-v2, step6-ft-reader-large-v2, ...) so the slug-based lookups in the
notebooks keep working unchanged.

Env overrides: NQ_ROOT, NQ_ARTIFACTS, NQ_RUN_ID (default: today's date, UTC).
"""
import datetime
import glob
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

# The Mac (Intel, no GPU) only ever does small inspection runs: 50 questions, no HF
# uploads, results in run_local50 (git-ignored). Set the variables to override.
IS_LOCAL = sys.platform == 'darwin'
if IS_LOCAL:
    os.environ.setdefault('NQ_N_QUESTIONS', '50')
    os.environ.setdefault('NQ_HF_PUSH', '0')
    os.environ.setdefault('NQ_RUN_ID', 'local50')

ROOT = Path(os.environ.get('NQ_ROOT', Path(__file__).resolve().parent.parent))
ART_DIR = Path(os.environ.get('NQ_ARTIFACTS', ROOT / 'artifacts'))
RUN_ID = os.environ.get('NQ_RUN_ID', datetime.datetime.now(datetime.timezone.utc).date().isoformat())
RESULTS_DIR = ROOT / 'results' / f'run_{RUN_ID}'
REQUIREMENTS = ROOT / ('requirements-local.txt' if IS_LOCAL else 'requirements.txt')

_T0 = {}


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def art(name):
    """Directory for one artifact set (old Kaggle dataset slug), created on demand."""
    p = ART_DIR / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_artifact(fname):
    """First file named `fname` anywhere under the artifacts tree."""
    hits = sorted(glob.glob(str(ART_DIR / '**' / fname), recursive=True))
    if not hits:
        raise FileNotFoundError(f'{fname} not found under {ART_DIR} - run the notebook that produces it first.')
    return hits[0]


def find_model_dir(keyword):
    """Directory of the first model.safetensors whose path contains `keyword`."""
    for p in sorted(glob.glob(str(ART_DIR / '**' / 'model.safetensors'), recursive=True)):
        if keyword in p.lower():
            return os.path.dirname(p)
    raise FileNotFoundError(f'no model.safetensors under {ART_DIR} in a path containing "{keyword}"')


# Artifacts live in a private HF dataset repo. Set NQ_HF_REPO to your own before running.
HF_REPO = os.environ.get('NQ_HF_REPO', '<your-hf-user>/nq-open-rag-snapshot')


def hf_push(local_path, path_in_repo, message):
    """Upload a file or a folder to the private HF dataset repo; returns the commit id.

    Needs an HF token (HF_TOKEN env var on the pod, `huggingface-cli login` on the Mac).
    Without one it prints a warning and uploads nothing, so a run never dies on it.
    """
    from huggingface_hub import HfApi, get_token
    if os.environ.get('NQ_HF_PUSH', '1') == '0':          # local / inspection runs
        print(f'NQ_HF_PUSH=0 - {local_path} not uploaded')
        return None
    if not (os.environ.get('HF_TOKEN') or get_token()):
        print(f'WARNING: no HF token - {local_path} NOT uploaded')
        return None
    api, p = HfApi(), Path(local_path)
    if p.is_dir():
        info = api.upload_folder(repo_id=HF_REPO, repo_type='dataset', folder_path=str(p),
                                 path_in_repo=path_in_repo, commit_message=message)
    else:
        info = api.upload_file(repo_id=HF_REPO, repo_type='dataset', path_or_fileobj=str(p),
                               path_in_repo=path_in_repo, commit_message=message)
    print(f'uploaded {local_path} -> hf://datasets/{HF_REPO}/{path_in_repo}  (commit {info.oid[:8]})')
    return info.oid


def hf_pull(path_in_repo):
    """Download one file from the HF dataset repo into ART_DIR/<path_in_repo>; returns its path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO, repo_type='dataset', filename=path_in_repo,
                           local_dir=str(ART_DIR))


def seed_everything(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            return {'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    'torch': torch.__version__, 'cuda': torch.version.cuda}
        return {'gpus': [], 'torch': torch.__version__, 'cuda': None}
    except ImportError:
        return {'gpus': [], 'torch': None, 'cuda': None}


def _write_merged(path, update):
    data = json.load(open(path)) if path.exists() else {}
    data.update(update)
    json.dump(data, open(path, 'w'), indent=2, default=str)
    return path


def start(nb, config):
    """Log the environment and CONFIG for notebook `nb`; start its wall clock."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _T0[nb] = time.time()
    try:
        freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                                capture_output=True, text=True, timeout=120).stdout.splitlines()
    except Exception as e:                       # never let logging kill a run
        freeze = [f'pip freeze failed: {type(e).__name__}: {e}']
    env = {'notebook': nb, 'run_id': RUN_ID, 'started_utc': _utc(),
           'python': sys.version.split()[0], 'platform': platform.platform(),
           **gpu_info(), 'config': config, 'pip_freeze': freeze}
    path = RESULTS_DIR / f'{nb}_env.json'
    json.dump(env, open(path, 'w'), indent=2, default=str)
    print(f'run {RUN_ID} | {nb} | gpus={env["gpus"]} | env -> {path}')
    return path


def log_metrics(nb, **metrics):
    """Merge metrics into results/run_<RUN_ID>/<nb>_metrics.json; stamps elapsed time."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = {'updated_utc': _utc()}
    if nb in _T0:
        stamp['elapsed_s'] = round(time.time() - _T0[nb], 1)
    path = _write_merged(RESULTS_DIR / f'{nb}_metrics.json', {**metrics, **stamp})
    print(f'metrics -> {path}  ({", ".join(metrics)})')
    return path
