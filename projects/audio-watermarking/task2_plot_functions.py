#!/usr/bin/env python3
"""
Plot watermark evidence for Task 2 clusters.

Produces PNGs under `outputs/plots/`:
 - waveform, averaged spectrum, envelope spectrum, and spectrogram for each representative file
 - one combined plot per representative file (for easy inclusion in reports)

Usage:
    python task2_plot_functions.py
"""
import os
import glob
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert, spectrogram
from scipy.fft import rfft, rfftfreq
from scipy.cluster.vq import kmeans2

BASE = os.path.dirname(__file__)
TASK2_DIR = os.path.join(BASE, 'ExIn', 'Task 2')
OUT_DIR = os.path.join(BASE, 'outputs', 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-16


def ensure_mono(sig):
    if sig.ndim == 1:
        return sig
    return np.mean(sig, axis=1)


def compute_basic_features(path):
    sig, sr = sf.read(path, dtype='float32')
    mono = ensure_mono(sig)
    n = len(mono)
    mean = float(np.mean(mono))
    rms = float(np.sqrt(np.mean(mono**2) + EPS))
    N = min(65536, n)
    block = mono[:N]
    freqs = rfftfreq(N, 1.0/sr)
    spec = np.abs(rfft(block))
    total_energy = np.sum(spec) + EPS
    hf_mask = freqs > 12000
    hf_ratio = float(np.sum(spec[hf_mask]) / total_energy)
    mask = freqs > 20
    peaks, props = find_peaks(spec[mask], height=np.max(spec[mask]) * 0.1)
    if peaks.size > 0:
        idx = peaks[np.argmax(props['peak_heights'])]
        top_freq = float(freqs[mask][idx])
    else:
        idx = np.argmax(spec[mask])
        top_freq = float(freqs[mask][idx])
    return {
        'path': path,
        'sr': sr,
        'mean': mean,
        'rms': rms,
        'hf_ratio': hf_ratio,
        'top_freq': top_freq,
    }


def build_matrix(feats_list):
    X = []
    for f in feats_list:
        norm_top_f = f['top_freq'] / max(1.0, f['sr']/2.0)
        X.append([f['mean'], f['rms'], f['hf_ratio'], norm_top_f])
    return np.array(X, dtype=float)


def pick_three_nearest(X, centroids, labels):
    groups = []
    for k, c in enumerate(centroids):
        idxs = np.where(labels == k)[0]
        if idxs.size == 0:
            groups.append([])
            continue
        dists = np.linalg.norm(X[idxs] - c[None, :], axis=1)
        order = np.argsort(dists)
        chosen = idxs[order[:3]]
        groups.append(list(chosen))
    return groups


def plot_for_file(path, out_prefix):
    sig, sr = sf.read(path, dtype='float32')
    mono = ensure_mono(sig)
    t = np.arange(len(mono)) / float(sr)

    # waveform
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    axes[0].plot(t, mono, linewidth=0.5)
    axes[0].set_title(os.path.basename(path) + ' — waveform')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')

    # averaged spectrum (FFT of first block)
    N = min(65536, len(mono))
    block = mono[:N]
    spec = np.abs(rfft(block))
    freqs = rfftfreq(N, 1.0/sr)
    axes[1].semilogy(freqs, spec + EPS)
    axes[1].set_xlim(0, min(20000, freqs.max()))
    axes[1].set_title('Magnitude spectrum (first %d samples)' % N)
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_ylabel('Magnitude')

    # envelope spectrum (AM detection)
    analytic = hilbert(mono)
    env = np.abs(analytic)
    Nenv = min(65536, len(env))
    env_spec = np.abs(rfft(env[:Nenv]))
    env_freqs = rfftfreq(Nenv, 1.0/sr)
    axes[2].semilogy(env_freqs, env_spec + EPS)
    axes[2].set_xlim(0, min(50, env_freqs.max()))
    axes[2].set_title('Envelope spectrum (AM detection)')
    axes[2].set_xlabel('Frequency [Hz]')
    axes[2].set_ylabel('Magnitude')

    # spectrogram
    f, tt, Sxx = spectrogram(mono, fs=sr, nperseg=1024, noverlap=512)
    # reduce dynamic range plotting to keep image sizes reasonable
    img = 10 * np.log10(Sxx + EPS)
    axes[3].pcolormesh(tt, f, img, shading='gouraud')
    axes[3].set_ylim(0, min(16000, f.max()))
    axes[3].set_xlabel('Time [s]')
    axes[3].set_ylabel('Frequency [Hz]')
    axes[3].set_title('Spectrogram')

    plt.tight_layout()
    outpng = out_prefix + '.png'
    fig.savefig(outpng, dpi=150)
    plt.close(fig)

    return outpng


def main():
    files = sorted(glob.glob(os.path.join(TASK2_DIR, '*_watermarked.wav')))
    if not files:
        print('No files found in', TASK2_DIR)
        return
    feats = [compute_basic_features(p) for p in files]
    X = build_matrix(feats)
    centroids, labels = kmeans2(X, 3, minit='points')
    groups_idx = pick_three_nearest(X, centroids, labels)

    created = []
    for k, idxs in enumerate(groups_idx):
        group_dir = os.path.join(OUT_DIR, f'group_{k+1}')
        os.makedirs(group_dir, exist_ok=True)
        sel_files = [files[i] for i in idxs] if idxs else []
        for i, p in enumerate(sel_files):
            out_prefix = os.path.join(group_dir, os.path.basename(p).replace('.wav', ''))
            print('Plotting', os.path.basename(p), '->', out_prefix + '.png')
            try:
                png = plot_for_file(p, out_prefix)
                created.append(png)
            except Exception as e:
                print('  Failed to plot', os.path.basename(p), ':', e)

    if not created:
        print('No plots created (no representative files).')
    else:
        print('Created plots:')
        for c in created:
            print(' -', c)

if __name__ == '__main__':
    main()
