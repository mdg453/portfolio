#!/usr/bin/env python3
"""
Extract and display simple watermark functions using 3 representative files per cluster.

Approach:
 - Reuse feature extraction and k-means clustering from `task2_detector.py`.
 - For each cluster, pick the 3 files closest to the cluster centroid.
 - For those 3 files, estimate:
    - DC offset (mean)
    - Dominant tone frequency (from averaged spectrum)
    - AM modulation frequency (from envelope FFT)
    - Tone amplitude and modulation depth estimate
 - Write a human-readable report to `outputs/task2_extracted_functions.txt`.
"""
import os
import glob
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, hilbert, butter, filtfilt
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


def compute_basic_features(path):
    sig, sr = sf.read(path, dtype='float32')
    mono = ensure_mono(sig)
    n = len(mono)
    mean = float(np.mean(mono))
    rms = float(np.sqrt(np.mean(mono**2) + EPS))
    # short FFT for features
    N = min(65536, n)
    block = mono[:N]
    freqs = rfftfreq(N, 1.0/sr)
    spec = np.abs(rfft(block))
    total_energy = np.sum(spec) + EPS
    hf_mask = freqs > 12000
    hf_ratio = float(np.sum(spec[hf_mask]) / total_energy)
    # top freq
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
        # indices of points in cluster k
        idxs = np.where(labels == k)[0]
        if idxs.size == 0:
            groups.append([])
            continue
        # compute distances to centroid
        dists = np.linalg.norm(X[idxs] - c[None, :], axis=1)
        # sort and pick up to 3 nearest
        order = np.argsort(dists)
        chosen = idxs[order[:3]]
        groups.append(list(chosen))
    return groups


def bandpass(x, sr, low, high, order=6):
    nyq = 0.5 * sr
    lown = max(low/nyq, 1e-6)
    highn = min(high/nyq, 0.999999)
    b, a = butter(order, [lown := lown, highn], btype='band')
    return filtfilt(b, a, x)


def estimate_watermark_for_files(file_paths):
    # Load files, compute averaged spectrum and envelope spectrum
    sigs = []
    srs = []
    means = []
    for p in file_paths:
        s, sr = sf.read(p, dtype='float32')
        mono = ensure_mono(s)
        sigs.append(mono)
        srs.append(sr)
        means.append(float(np.mean(mono)))
    # assume same sr
    sr = srs[0]
    # pad/crop to shortest length for averaging
    minlen = min(len(s) for s in sigs)
    trimmed = np.array([s[:minlen] for s in sigs])
    avg_sig = np.mean(trimmed, axis=0)
    # DC offset estimate: mean across averaged signal
    dc_est = float(np.mean(avg_sig))
    # Spectrum of average
    N = min(131072, minlen)
    spec = np.abs(rfft(avg_sig[:N]))
    freqs = rfftfreq(N, 1.0/sr)
    # find top peaks above 20 Hz
    mask = freqs > 20
    specm = spec[mask]
    freqs_m = freqs[mask]
    peaks, props = find_peaks(specm, height=np.max(specm)*0.05)
    peak_freqs = freqs_m[peaks] if peaks.size>0 else freqs_m[np.argsort(specm)[-5:]]
    peak_freqs = np.sort(peak_freqs)
    # choose the strongest peak as tone candidate
    if peaks.size > 0:
        strongest_idx = np.argmax(props['peak_heights'])
        tone_freq = float(peak_freqs[strongest_idx])
    else:
        tone_freq = float(peak_freqs[-1]) if peak_freqs.size>0 else 0.0
    # If tone_freq is small or unreliable, set tone_freq=0
    if tone_freq < 30.0:
        # may still be valid low-freq tones
        pass
    # Bandpass around tone and compute envelope
    if tone_freq > 0.0:
        bw = max(10.0, 0.05 * tone_freq)
        low = max(1.0, tone_freq - bw)
        high = min(sr/2.0 - 10.0, tone_freq + bw)
        try:
            tone_component = bandpass(avg_sig, sr, low, high)
            env = np.abs(hilbert(tone_component))
            Nenv = min(65536, len(env))
            env_spec = np.abs(rfft(env[:Nenv]))
            env_freqs = rfftfreq(Nenv, 1.0/sr)
            # look for envelope peaks > 0.5 Hz
            env_mask = env_freqs > 0.5
            if np.any(env_mask):
                env_m = env_spec[env_mask]
                ef = env_freqs[env_mask]
                epeaks, eprops = find_peaks(env_m, height=np.max(env_m)*0.1)
                if epeaks.size>0:
                    env_peak_freq = float(ef[epeaks[np.argmax(eprops['peak_heights'])]])
                    env_peak_mag = float(np.max(eprops['peak_heights']))
                else:
                    env_peak_freq = 0.0
                    env_peak_mag = 0.0
            else:
                env_peak_freq = 0.0
                env_peak_mag = 0.0
            # tone amplitude estimate (RMS)
            tone_rms = float(np.sqrt(np.mean(tone_component**2) + EPS))
            # modulation depth approx: (max(env)-min(env))/mean(env)
            if np.mean(env) > 0:
                mod_depth = float((np.max(env) - np.min(env)) / (np.mean(env) + EPS))
            else:
                mod_depth = 0.0
        except Exception:
            env_peak_freq = 0.0
            env_peak_mag = 0.0
            tone_rms = 0.0
            mod_depth = 0.0
    else:
        env_peak_freq = 0.0
        env_peak_mag = 0.0
        tone_rms = 0.0
        mod_depth = 0.0

    # HF PN indicator
    # compute HF energy ratio across the 3 files
    hf_ratios = []
    for s in trimmed:
        Nf = min(65536, len(s))
        sp = np.abs(rfft(s[:Nf]))
        fq = rfftfreq(Nf, 1.0/sr)
        hf_mask = fq > 12000
        hf_ratios.append(float(np.sum(sp[hf_mask]) / (np.sum(sp)+EPS)))
    hf_avg = float(np.mean(hf_ratios))

    result = {
        'dc_est': dc_est,
        'tone_freq': tone_freq,
        'env_peak_freq': env_peak_freq,
        'tone_rms': tone_rms,
        'mod_depth': mod_depth,
        'hf_avg': hf_avg,
    }
    return result


def main():
    files = sorted(glob.glob(os.path.join(TASK2_DIR, '*_watermarked.wav')))
    if len(files) < 9:
        print('Expected 9 files in Task 2 directory; found', len(files))
    feats = [compute_basic_features(p) for p in files]
    X = build_matrix(feats)
    centroids, labels = kmeans2(X, 3, minit='points')
    groups_idx = pick_three_nearest(X, centroids, labels)

    lines = []
    lines.append('Task 2 — extracted watermark functions (3 representative files per cluster)')
    for k, idxs in enumerate(groups_idx):
        lines.append('\nGroup %d:' % (k+1))
        sel_files = [files[i] for i in idxs] if idxs else []
        if not sel_files:
            lines.append('  (no files selected)')
            continue
        for p in sel_files:
            lines.append('  - ' + os.path.basename(p))
        res = estimate_watermark_for_files(sel_files)
        # format function guess
        lines.append('  Estimated watermark parameters:')
        lines.append(f"    DC offset (approx): {res['dc_est']:.6g}")
        if res['tone_freq'] > 0.0 and res['tone_rms'] > 1e-6:
            lines.append(f"    Tone freq (Hz): {res['tone_freq']:.2f}")
            lines.append(f"    Tone RMS amplitude: {res['tone_rms']:.6g}")
            lines.append(f"    AM freq (Hz): {res['env_peak_freq']:.3f}")
            lines.append(f"    Estimated modulation depth (rel.): {res['mod_depth']:.3f}")
            lines.append(f"    HF energy ratio (avg): {res['hf_avg']:.6f}")
            lines.append('  Function form: add A*(1 + m*sin(2π·f_m·t)) * sin(2π·f_0·t)  (A and m above)')
        else:
            lines.append(f"    No clear tone detected. HF energy ratio (avg): {res['hf_avg']:.6f}")
            # if DC offset significant
            if abs(res['dc_est']) > 1e-4:
                lines.append('  Function form: add constant DC offset c (x[n] -> x[n] + c)')

    outp = os.path.join(OUT_DIR, 'task2_extracted_functions.txt')
    with open(outp, 'w', encoding='utf8') as fh:
        fh.write('\n'.join(lines))
    print('\n'.join(lines))
    print('\nWrote detailed extraction to', outp)

if __name__ == '__main__':
    main()
