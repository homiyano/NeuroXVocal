"""
Evaluate NeuroXVocal on the ADReSSo diagnosis-test set.

Usage:
    python evaluate_test_set.py \
        --test_audio_dir /path/to/diagnosis-test/diagnosis/test-dist/audio \
        --model_path /path/to/results/best.pth \
        --output_csv /path/to/predictions.csv \
        [--labels_csv /path/to/ground_truth.csv]  # optional, for accuracy metrics

Ground truth CSV format (if available):
    patient_id,label
    adrsdt1,1
    adrsdt2,0
    ...  (1=AD, 0=CN)
"""

import os
import sys
import argparse
import tempfile
import shutil
import subprocess
import joblib
import re
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Add train dir to path for model import
SCRIPT_DIR  = Path(__file__).parent
TRAIN_DIR   = SCRIPT_DIR.parent / 'train'
PROC_DIR    = SCRIPT_DIR.parent / 'data_processing'
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(PROC_DIR))

from models import NeuroXVocal

TEXT_EMBEDDING_MODEL   = 'microsoft/deberta-v3-base'
NUM_MFCC_FEATURES      = 47
NUM_EMBEDDING_FEATURES = 768
SCALER_FEATURES        = SCRIPT_DIR / 'scaler_params_audio_features.pkl'
SCALER_EMB             = SCRIPT_DIR / 'scaler_params_audio_emb.pkl'


# ── Pipeline steps ───────────────────────────────────────────────────

def transcribe(wav_path, whisper_model):
    result = whisper_model.transcribe(str(wav_path), fp16=False)
    return result['text']


def preprocess_text(text):
    text = text.lower()
    text = ' '.join(text.split())
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text


def extract_audio_features(wav_path):
    extract_script = SCRIPT_DIR.parent / 'data_extraction' / 'extract_audio_features.py'
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_csv = tmp_dir / 'features.csv'
    subprocess.run(
        [sys.executable, str(extract_script), str(wav_path.parent),
         '--output_csv', str(tmp_csv)],
        capture_output=True, check=True
    )
    df = pd.read_csv(tmp_csv)
    # keep only the row for this specific file
    patient_id = wav_path.stem
    row = df[df['patient_id'] == patient_id]
    shutil.rmtree(tmp_dir)
    return row


def extract_audio_embeddings(wav_path, emb_model, processor, device):
    speech, sr = sf.read(str(wav_path), always_2d=True)
    speech = speech.mean(axis=1).astype(np.float32)  # stereo -> mono
    if sr != 16000:
        g = gcd(sr, 16000)
        speech = resample_poly(speech, 16000 // g, sr // g).astype(np.float32)
    inputs = processor(speech, sampling_rate=16000,
                       return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = emb_model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb


def preprocess_features(df_row, scaler_path, drop_cols):
    df = df_row.copy()
    patient_id = df['patient_id'].values
    df = df.drop(columns=['patient_id'] + drop_cols, errors='ignore')
    df = df.apply(lambda x: x.fillna(x.mean()) if x.isnull().any() else x)
    scaler = joblib.load(scaler_path)
    scaled = scaler.transform(df)
    df_out = pd.DataFrame(scaled, columns=df.columns)
    df_out.insert(0, 'patient_id', patient_id)
    return df_out


def preprocess_embeddings(emb_array, scaler_path):
    scaler = joblib.load(scaler_path)
    scaled = scaler.transform(emb_array.reshape(1, -1))
    return scaled.flatten()


# ── Inference ────────────────────────────────────────────────────────

def load_model(model_path, device):
    model = NeuroXVocal(
        num_audio_features=NUM_MFCC_FEATURES,
        num_embedding_features=NUM_EMBEDDING_FEATURES,
        text_embedding_model=TEXT_EMBEDDING_MODEL,
    )
    state_dict = torch.load(model_path, map_location=device)
    if 'module.' in list(state_dict.keys())[0]:
        from collections import OrderedDict
        state_dict = OrderedDict(
            (k.replace('module.', ''), v) for k, v in state_dict.items()
        )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(model, tokenizer, text, audio_features_row, emb_array, device):
    # Text
    tokens = tokenizer(text, padding='max_length', truncation=True,
                       max_length=512, return_tensors='pt')
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # Audio features (drop patient_id)
    af = audio_features_row.drop(columns=['patient_id']).iloc[0].values.astype(float)
    af_tensor = torch.tensor(af, dtype=torch.float32).unsqueeze(0).to(device)

    # Embeddings
    emb_tensor = torch.tensor(emb_array, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tokens, af_tensor, emb_tensor)
        confidence = torch.sigmoid(out).item()
        predicted = 1 if confidence > 0.5 else 0

    return predicted, confidence


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_audio_dir', required=True,
                        help='Path to diagnosis-test/diagnosis/test-dist/audio/')
    parser.add_argument('--model_path', required=True,
                        help='Path to best.pth')
    parser.add_argument('--output_csv', default='predictions.csv',
                        help='Where to save predictions')
    parser.add_argument('--labels_csv', default=None,
                        help='Optional CSV with ground truth (patient_id, label)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    wav_files = sorted(Path(args.test_audio_dir).glob('*.wav'))
    print(f'Found {len(wav_files)} test files\n')

    print('Loading models...')
    whisper_model = whisper.load_model('base')
    tokenizer     = AutoTokenizer.from_pretrained(TEXT_EMBEDDING_MODEL)
    emb_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    emb_model     = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').to(device).eval()
    classifier    = load_model(args.model_path, device)
    print('All models loaded.\n')

    # Columns to drop from audio features before scaling
    drop_cols = ['jitter_local', 'shimmer_local', 'formant_1_mean', 'formant_1_std',
                 'formant_2_mean', 'formant_2_std', 'formant_3_mean', 'formant_3_std', 'class']

    results = []
    for wav in wav_files:
        patient_id = wav.stem
        print(f'Processing {patient_id}...')
        try:
            # 1. Transcribe
            text = transcribe(wav, whisper_model)
            text = preprocess_text(text)

            # 2. Audio features
            feat_row = extract_audio_features(wav)
            if feat_row.empty:
                print(f'  WARNING: no features extracted for {patient_id}, skipping')
                continue
            feat_proc = preprocess_features(feat_row, SCALER_FEATURES, drop_cols)

            # 3. Audio embeddings
            emb_raw  = extract_audio_embeddings(wav, emb_model, emb_processor, device)
            emb_proc = preprocess_embeddings(emb_raw, SCALER_EMB)

            # 4. Predict
            pred, conf = predict(classifier, tokenizer, text, feat_proc, emb_proc, device)
            label_str  = 'AD' if pred == 1 else 'CN'
            print(f'  → {label_str}  (confidence: {conf:.4f})')
            results.append({
                'patient_id':  patient_id,
                'prediction':  pred,
                'label':       label_str,
                'confidence':  round(conf, 4),
                'transcription': text[:120] + '...' if len(text) > 120 else text
            })
        except Exception as e:
            print(f'  ERROR: {e}')
            results.append({'patient_id': patient_id, 'prediction': None,
                            'label': 'ERROR', 'confidence': None, 'transcription': str(e)})

    # Save predictions
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output_csv, index=False)
    print(f'\nPredictions saved to {args.output_csv}')
    print(df_results[['patient_id', 'label', 'confidence']].to_string(index=False))

    # Evaluate if ground truth provided
    if args.labels_csv:
        gt = pd.read_csv(args.labels_csv)
        merged = df_results.merge(gt, on='patient_id', suffixes=('_pred', '_true'))
        merged = merged.dropna(subset=['prediction'])

        y_true = merged['label_true'].values
        y_pred = merged['prediction'].values

        acc = accuracy_score(y_true, y_pred)
        print(f'\n{"="*50}')
        print(f'Test Set Accuracy: {acc*100:.2f}%  ({int(acc*len(y_true))}/{len(y_true)})')
        print('='*50)
        print(classification_report(y_true, y_pred, target_names=['CN (0)', 'AD (1)']))
        print('Confusion Matrix:')
        print(confusion_matrix(y_true, y_pred))


if __name__ == '__main__':
    main()
