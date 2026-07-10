"""
NeuroXVocal — Inference with VRAM & Timing Monitor

Tracks peak/min VRAM and wall-clock time for each of the 3 inference pipelines:
  Pipeline 1 — Whisper transcription
  Pipeline 2 — Wav2Vec2 audio embedding extraction
  Pipeline 3 — NeuroXVocal classification (DeBERTa + classifier)

Also reports model load memory cost.

Usage:
    python evaluate_monitor.py \
        --test_audio_dir /path/to/diagnosis-test/diagnosis/test-dist/audio \
        --model_path /path/to/results/best.pth \
        [--max_patients 5]   # limit to N patients for a quick test
"""

import os
import sys
import argparse
import tempfile
import shutil
import subprocess
import joblib
import re
import time
import warnings
import torch
import soundfile as sf
import numpy as np
import whisper
import pandas as pd
from pathlib import Path
from scipy.signal import resample_poly
from math import gcd
from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

SCRIPT_DIR = Path(__file__).parent
TRAIN_DIR  = SCRIPT_DIR.parent / 'train'
PROC_DIR   = SCRIPT_DIR.parent / 'data_processing'
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(PROC_DIR))

from models import NeuroXVocal

TEXT_EMBEDDING_MODEL   = 'microsoft/deberta-v3-base'
NUM_MFCC_FEATURES      = 47
NUM_EMBEDDING_FEATURES = 768
SCALER_FEATURES        = SCRIPT_DIR / 'scaler_params_audio_features.pkl'
SCALER_EMB             = SCRIPT_DIR / 'scaler_params_audio_emb.pkl'
DROP_COLS              = ['jitter_local', 'shimmer_local', 'formant_1_mean', 'formant_1_std',
                          'formant_2_mean', 'formant_2_std', 'formant_3_mean', 'formant_3_std', 'class']


# ── VRAM Monitor ─────────────────────────────────────────────────────────────

class VRAMMonitor:
    """Tracks VRAM (CUDA or MPS) and wall-clock time."""

    def __init__(self, device: torch.device):
        self.device      = device
        self.dtype       = device.type   # 'cuda', 'mps', or 'cpu'
        self._global_peak = 0.0

    # ── raw reads ──────────────────────────────────────────────────────────

    def current_mb(self) -> float:
        if self.dtype == 'cuda':
            return torch.cuda.memory_allocated(self.device) / 1024 ** 2
        if self.dtype == 'mps':
            return torch.mps.current_allocated_memory() / 1024 ** 2
        return 0.0

    def peak_mb(self) -> float:
        if self.dtype == 'cuda':
            return torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
        if self.dtype == 'mps':
            return torch.mps.driver_allocated_memory() / 1024 ** 2
        return 0.0

    def reset_peak(self):
        if self.dtype == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

    def sync(self):
        if self.dtype == 'cuda':
            torch.cuda.synchronize(self.device)
        elif self.dtype == 'mps':
            torch.mps.synchronize()

    # ── context manager for one pipeline stage ─────────────────────────────

    def measure(self, label: str) -> '_Stage':
        return _Stage(self, label)

    def update_global_peak(self, peak: float):
        if peak > self._global_peak:
            self._global_peak = peak

    def global_peak_mb(self) -> float:
        return self._global_peak


class _Stage:
    """Context manager: wraps one pipeline step, captures VRAM delta + time."""

    def __init__(self, mon: VRAMMonitor, label: str):
        self.mon   = mon
        self.label = label
        self.vram_before = self.vram_after = self.peak = self.elapsed = 0.0

    def __enter__(self):
        self.mon.sync()
        self.mon.reset_peak()
        self.vram_before = self.mon.current_mb()
        self._t0         = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.mon.sync()
        self.elapsed    = time.perf_counter() - self._t0
        self.vram_after = self.mon.current_mb()
        self.peak       = self.mon.peak_mb()
        self.mon.update_global_peak(self.peak)


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def transcribe(wav_path, whisper_model):
    result = whisper_model.transcribe(str(wav_path), fp16=False)
    text   = result['text'].lower()
    return re.sub(r'[^a-zA-Z0-9\s.,!?]', '', ' '.join(text.split()))


def extract_audio_features(wav_path):
    script  = SCRIPT_DIR.parent / 'data_extraction' / 'extract_audio_features.py'
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_csv = tmp_dir / 'features.csv'
    subprocess.run(
        [sys.executable, str(script), str(wav_path.parent),
         '--output_csv', str(tmp_csv)],
        capture_output=True, check=True
    )
    df  = pd.read_csv(tmp_csv)
    row = df[df['patient_id'] == wav_path.stem]
    shutil.rmtree(tmp_dir)
    return row


def preprocess_features(df_row):
    df  = df_row.copy()
    pid = df['patient_id'].values
    df  = df.drop(columns=['patient_id'] + DROP_COLS, errors='ignore')
    df  = df.apply(lambda x: x.fillna(x.mean()) if x.isnull().any() else x)
    scaled = joblib.load(SCALER_FEATURES).transform(df)
    out = pd.DataFrame(scaled, columns=df.columns)
    out.insert(0, 'patient_id', pid)
    return out


def extract_audio_embeddings(wav_path, emb_model, processor, device):
    speech, sr = sf.read(str(wav_path), always_2d=True)
    speech     = speech.mean(axis=1).astype(np.float32)
    if sr != 16000:
        g      = gcd(sr, 16000)
        speech = resample_poly(speech, 16000 // g, sr // g).astype(np.float32)
    inputs = processor(speech, sampling_rate=16000, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = emb_model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    scaled = joblib.load(SCALER_EMB).transform(emb.reshape(1, -1)).flatten()
    return scaled


def predict(model, tokenizer, text, feat_row, emb_array, device):
    tokens = tokenizer(text, padding='max_length', truncation=True,
                       max_length=512, return_tensors='pt')
    tokens = {k: v.to(device) for k, v in tokens.items()}
    af     = feat_row.drop(columns=['patient_id']).iloc[0].values.astype(float)
    af_t   = torch.tensor(af,        dtype=torch.float32).unsqueeze(0).to(device)
    emb_t  = torch.tensor(emb_array, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        conf = torch.sigmoid(model(tokens, af_t, emb_t)).item()
    return (1 if conf > 0.5 else 0), conf


def load_classifier(model_path, device):
    model = NeuroXVocal(
        num_audio_features=NUM_MFCC_FEATURES,
        num_embedding_features=NUM_EMBEDDING_FEATURES,
        text_embedding_model=TEXT_EMBEDDING_MODEL,
    )
    sd = torch.load(model_path, map_location=device)
    if 'module.' in list(sd.keys())[0]:
        from collections import OrderedDict
        sd = OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


# ── Reporting ─────────────────────────────────────────────────────────────────

def _mb(v): return f'{v:>8.1f} MB'
def _sec(v): return f'{v:>6.2f} s'


def print_model_load_table(load_records):
    print('\n' + '=' * 62)
    print('  MODEL LOAD — VRAM COST')
    print('=' * 62)
    print(f'  {"Model":<22}  {"Before":>9}  {"After":>9}  {"Delta":>9}')
    print('-' * 62)
    for r in load_records:
        delta = r['after'] - r['before']
        print(f'  {r["name"]:<22}  {_mb(r["before"])}  {_mb(r["after"])}  {_mb(delta)}')
    print('=' * 62)


def print_pipeline_table(pipeline_records):
    print('\n' + '=' * 74)
    print('  PER-PATIENT PIPELINE BREAKDOWN')
    print('=' * 74)
    hdr = f'  {"Patient":<12}  {"Pipeline":<28}  {"Peak VRAM":>9}  {"Time":>7}'
    print(hdr)
    print('-' * 74)
    prev_pid = None
    for r in pipeline_records:
        sep = '  ' if r['patient'] == prev_pid else '\n  '
        prev_pid = r['patient']
        print(f'{sep}{r["patient"]:<12}  {r["pipeline"]:<28}  {_mb(r["peak"])}  {_sec(r["elapsed"])}')
    print('=' * 74)


def print_summary_table(pipeline_records, global_peak, device_type):
    df = pd.DataFrame(pipeline_records)
    print('\n' + '=' * 62)
    print('  SUMMARY — per pipeline (across all patients)')
    print('=' * 62)
    print(f'  {"Pipeline":<28}  {"Peak max":>9}  {"Peak min":>9}  {"Avg time":>8}')
    print('-' * 62)
    for pipe, grp in df.groupby('pipeline', sort=False):
        print(f'  {pipe:<28}  {_mb(grp["peak"].max())}  {_mb(grp["peak"].min())}  {_sec(grp["elapsed"].mean())}')
    print('-' * 62)
    label = 'CUDA' if device_type == 'cuda' else ('MPS' if device_type == 'mps' else 'CPU')
    print(f'  {"Global peak VRAM (" + label + ")":<42}  {_mb(global_peak)}')
    print(f'  {"Global min VRAM (baseline before load)":<42}  {_mb(df["peak"].min())}')
    print('=' * 62)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_audio_dir', required=True)
    parser.add_argument('--model_path',     required=True)
    parser.add_argument('--max_patients',   type=int, default=None,
                        help='Limit to N patients (for a quick test run)')
    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        ('mps' if torch.backends.mps.is_available() else 'cpu')
    )
    print(f'\nDevice : {device}  ({device.type.upper()})')
    mon = VRAMMonitor(device)

    # ── Load models, measure VRAM cost of each ──────────────────────────────

    load_records = []

    def load_and_record(name, fn):
        mon.sync()
        before = mon.current_mb()
        obj    = fn()
        mon.sync()
        after  = mon.current_mb()
        load_records.append({'name': name, 'before': before, 'after': after})
        print(f'  Loaded {name:<22}  +{after - before:.1f} MB  (total {after:.1f} MB)')
        return obj

    print('\n── Loading models ──────────────────────────────────────')
    whisper_model = load_and_record('Whisper base',
                                    lambda: whisper.load_model('base'))
    tokenizer     = load_and_record('DeBERTa tokenizer',
                                    lambda: AutoTokenizer.from_pretrained(TEXT_EMBEDDING_MODEL))
    emb_processor = load_and_record('Wav2Vec2 processor',
                                    lambda: Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h'))
    emb_model     = load_and_record('Wav2Vec2 model',
                                    lambda: Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').to(device).eval())
    classifier    = load_and_record('NeuroXVocal (DeBERTa)',
                                    lambda: load_classifier(args.model_path, device))

    print_model_load_table(load_records)

    # ── Per-patient inference with pipeline monitoring ───────────────────────

    wav_files = sorted(Path(args.test_audio_dir).glob('*.wav'))
    if args.max_patients:
        wav_files = wav_files[:args.max_patients]
    print(f'\nRunning inference on {len(wav_files)} patients...\n')

    pipeline_records = []

    for wav in wav_files:
        pid = wav.stem
        print(f'Patient: {pid}')

        # ── Pipeline 1: Whisper transcription ──────────────────────────────
        with mon.measure('1. Whisper transcription') as stage:
            text = transcribe(wav, whisper_model)
        pipeline_records.append({
            'patient':  pid,
            'pipeline': '1. Whisper transcription',
            'peak':     stage.peak,
            'elapsed':  stage.elapsed,
        })
        print(f'  [Whisper]   peak={stage.peak:.1f} MB  time={stage.elapsed:.2f}s')

        # ── Audio features (subprocess — not tracked in VRAM) ──────────────
        t0        = time.perf_counter()
        feat_row  = extract_audio_features(wav)
        feat_proc = preprocess_features(feat_row)
        feat_time = time.perf_counter() - t0
        print(f'  [AudioFeat] subprocess  time={feat_time:.2f}s  (separate process, VRAM N/A)')

        # ── Pipeline 2: Wav2Vec2 embedding ─────────────────────────────────
        with mon.measure('2. Wav2Vec2 embedding') as stage:
            emb_proc = extract_audio_embeddings(wav, emb_model, emb_processor, device)
        pipeline_records.append({
            'patient':  pid,
            'pipeline': '2. Wav2Vec2 embedding',
            'peak':     stage.peak,
            'elapsed':  stage.elapsed,
        })
        print(f'  [Wav2Vec2]  peak={stage.peak:.1f} MB  time={stage.elapsed:.2f}s')

        # ── Pipeline 3: NeuroXVocal classification ─────────────────────────
        with mon.measure('3. NeuroXVocal classify') as stage:
            pred, conf = predict(classifier, tokenizer, text, feat_proc, emb_proc, device)
        pipeline_records.append({
            'patient':  pid,
            'pipeline': '3. NeuroXVocal classify',
            'peak':     stage.peak,
            'elapsed':  stage.elapsed,
        })
        label = 'AD' if pred == 1 else 'CN'
        print(f'  [Classify]  peak={stage.peak:.1f} MB  time={stage.elapsed:.2f}s  → {label} ({conf:.4f})\n')

    # ── Final report ─────────────────────────────────────────────────────────
    print_pipeline_table(pipeline_records)
    print_summary_table(pipeline_records, mon.global_peak_mb(), device.type)

    if device.type == 'cpu':
        print('\n  Note: device=CPU — VRAM columns show 0 MB (no GPU).')
        print('  Run on a CUDA or Apple MPS device to see real VRAM numbers.')


if __name__ == '__main__':
    main()
