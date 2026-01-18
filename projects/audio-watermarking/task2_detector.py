#!/usr/bin/env python3
"""
Task 2 detector: Group 9 audio files (ExIn/Task 2) into 3 watermark types
and attempt to describe each watermark function.

Outputs:
 - prints grouping and hypotheses
 - writes `outputs/task2_classification.txt`

Usage:
    python task2_detector.py

"""
import os
import glob
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, hilbert
from scipy.fft import rfft, rfftfreq
from scipy.cluster.vq import kmeans2

BASE = os.path.dirname(__file__)
TASK2_DIR = os.path.join(BASE, 'ExIn', 'Task 2')
OUT_DIR = os.path.join(BASE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-16


def ensure_mono(sig):
    if sig.ndim == 1:
        return sig
    return np.mean(sig, axis=1)


def file_list():
    pattern = os.path.join(TASK2_DIR, '*_watermarked.wav')
    return sorted(glob.glob(pattern))


def compute_features(path):
    sig, sr = sf.read(path, dtype='float32')
    mono = ensure_mono(sig)
    n = len(mono)
    # basic stats
    mean = float(np.mean(mono))
    rms = float(np.sqrt(np.mean(mono**2) + EPS))
    peak = float(np.max(np.abs(mono)))
    # FFT
    N = min(65536, n)
    block = mono[:N]
    freqs = rfftfreq(N, 1.0/sr)
    spec = np.abs(rfft(block))
    total_energy = np.sum(spec) + EPS
    # high-frequency energy ratio (>12kHz)
    hf_mask = freqs > 12000
    hf_ratio = float(np.sum(spec[hf_mask]) / total_energy)
    # top peak frequency (ignore very low <20Hz)
    mask = freqs > 20
    peaks, props = find_peaks(spec[mask], height=np.max(spec[mask]) * 0.1)
    top_freq = None
    top_mag = 0.0
    if peaks.size > 0:
        idx = peaks[np.argmax(props['peak_heights'])]
        top_freq = float(freqs[mask][idx])
        top_mag = float(props['peak_heights'].max())
    else:
        # fallback: global max
        idx = np.argmax(spec[mask])
        top_freq = float(freqs[mask][idx])
        top_mag = float(spec[mask][idx])
    # envelope analysis (for AM watermarks)
    analytic = hilbert(mono)
    env = np.abs(analytic)
    Nenv = min(65536, len(env))
    env_spec = np.abs(rfft(env[:Nenv]))
    env_freqs = rfftfreq(Nenv, 1.0/sr)
    # ignore DC peak in envelope
    env_mask = env_freqs > 0.5
    if np.any(env_mask):
        env_peak_idx = np.argmax(env_spec[env_mask])
        env_peak_freq = float(env_freqs[env_mask][env_peak_idx])
        env_peak_mag = float(env_spec[env_mask][env_peak_idx])
    else:
        env_peak_freq = 0.0
        env_peak_mag = 0.0

    feats = {
        'path': path,
        'sr': sr,
        'mean': mean,
        'rms': rms,
        'peak': peak,
        'hf_ratio': hf_ratio,
        'top_freq': top_freq,
        'top_mag': top_mag,
        'env_peak_freq': env_peak_freq,
        'env_peak_mag': env_peak_mag,
    }
    return feats


def build_feature_matrix(feats_list):
    # choose numeric features for clustering
    X = []
    for f in feats_list:
        # normalize top_freq by sr/2
        norm_top_f = f['top_freq'] / max(1.0, f['sr']/2.0)
        norm_env_f = f['env_peak_freq'] / max(1.0, f['sr']/2.0)
        X.append([f['mean'], f['rms'], f['hf_ratio'], norm_top_f, f['env_peak_mag']])
    return np.array(X, dtype=float)


def describe_cluster(cluster_feats):
    # cluster_feats: list of feature dicts
    desc = []
    means = np.array([f['mean'] for f in cluster_feats])
    hf = np.array([f['hf_ratio'] for f in cluster_feats])
    topfs = np.array([f['top_freq'] for f in cluster_feats])
    topmags = np.array([f['top_mag'] for f in cluster_feats])
    env_freqs = np.array([f['env_peak_freq'] for f in cluster_feats])
    env_mags = np.array([f['env_peak_mag'] for f in cluster_feats])

    # DC offset check
    if np.mean(np.abs(means)) > 1e-3:
        desc.append(f"Non-zero DC mean (mean abs {np.mean(np.abs(means)):.4g}) -> likely constant offset added")
    # HF broadband check
    if np.mean(hf) > 0.02:
        desc.append(f"Elevated high-frequency energy (avg hf_ratio {np.mean(hf):.4f}) -> likely added HF noise/spread-spectrum watermark")
    # tone check
    # check if top frequencies are consistent and strong
    if np.std(topfs) < 50 and np.mean(topmags) > 1e3:
        desc.append(f"Consistent strong spectral peak at ~{np.mean(topfs):.1f} Hz -> likely added sine tone at that frequency")
    # amplitude modulation check
    if np.mean(env_mags) > 1e2 and np.mean(env_freqs) > 1.0:
        desc.append(f"Envelope shows modulation at ~{np.mean(env_freqs):.1f} Hz -> likely amplitude modulation: multiply by (1 + a*sin(2πft))")

    if not desc:
        desc = ["No obvious simple spectral/offset pattern detected; could be subtle PN or spread-spectrum watermark."]
    return desc


def main():
    files = file_list()
    if not files:
        print(f"No files found in {TASK2_DIR}")
        return
    print(f"Found {len(files)} files. Computing features...")
    feats_list = [compute_features(p) for p in files]
    for f in feats_list:
        print(f"{os.path.basename(f['path'])}: mean={f['mean']:.6g}, rms={f['rms']:.6g}, hf_ratio={f['hf_ratio']:.6g}, top_freq={f['top_freq']:.1f}, env_peak={f['env_peak_freq']:.2f}")

    X = build_feature_matrix(feats_list)
    # run kmeans with k=3
    centroids, labels = kmeans2(X, 3, minit='points')

    groups = {0: [], 1: [], 2: []}
    for lab, f in zip(labels, feats_list):
        groups[int(lab)].append(f)

    summary_lines = []
    summary_lines.append("Task 2 classification results")
    for lab in sorted(groups.keys()):
        group = groups[lab]
        summary_lines.append('\nGroup %d:' % lab)
        for f in group:
            summary_lines.append(' - ' + os.path.basename(f['path']))
        # describe the cluster
        desc = describe_cluster(group)
        summary_lines.append('Hypothesis:')
        for d in desc:
            summary_lines.append('  - ' + d)

    out_txt = os.path.join(OUT_DIR, 'task2_classification.txt')
    with open(out_txt, 'w', encoding='utf8') as fh:
        fh.write('\n'.join(summary_lines))

    print('\n'.join(summary_lines))
    print(f"\nWrote classification to {out_txt}")

if __name__ == '__main__':
    main()
