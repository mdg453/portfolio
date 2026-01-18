#!/usr/bin/env python3
"""
Task 3: Determine which of the two Task 3 files was sped up in the time (signal)
domain vs the frequency domain, and estimate the speed-up ratio x.

Approach:
 - Compute mean magnitude spectrum for both files (use a large FFT block).
 - Find prominent spectral peaks in each spectrum.
 - Compute all pairwise frequency ratios between peaks of file A and file B.
 - Find the dominant ratio (median of ratios within a small tolerance) and count inliers.
 - If many ratios cluster around a value != 1, the file with higher peak frequencies
   is likely the time-domain resampled (pitch-shifted) version; the other is frequency-domain processed.
 - Report estimated x as the median ratio.

Usage:
    python task3_speedup.py

Outputs:
 - prints a short report and writes `outputs/task3_results.txt` with evidence.
"""

import os
import numpy as np
import glob
import soundfile as sf
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

BASE = os.path.dirname(__file__)
TASK3_DIR = os.path.join(BASE, 'ExIn', 'Task 3')
OUT_DIR = os.path.join(BASE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

def ensure_mono(sig):
    if sig.ndim == 1:
        return sig
    return np.mean(sig, axis=1)


def mean_spectrum(path, Nfft=131072):
    sig, sr = sf.read(path, dtype='float32')
    mono = ensure_mono(sig)
    # if signal shorter than Nfft, reduce Nfft
    N = min(Nfft, len(mono))
    block = mono[:N]
    spec = np.abs(rfft(block))
    freqs = rfftfreq(N, 1.0/sr)
    return freqs, spec, sr


def top_peaks(freqs, spec, topk=30, min_freq=20.0):
    mask = freqs >= min_freq
    f = freqs[mask]
    s = spec[mask]
    # find peaks with minimum prominence relative to max
    peaks, props = find_peaks(s, height=np.max(s)*0.02)
    if peaks.size == 0:
        # fallback: take topk by magnitude
        idxs = np.argsort(s)[-topk:]
        return np.sort(f[idxs])
    peak_freqs = f[peaks]
    # sort by magnitude and take topk
    heights = props['peak_heights']
    idx_order = np.argsort(heights)[-topk:]
    sel = np.sort(peak_freqs[idx_order])
    return sel


def estimate_ratio(peaks_a, peaks_b):
    # compute all pairwise ratios a/b
    ratios = []
    for fa in peaks_a:
        for fb in peaks_b:
            if fb <= 0 or fa <= 0:
                continue
            r = fa / fb
            # keep only reasonable ratios
            if 0.5 <= r <= 2.0:
                ratios.append(r)
    if not ratios:
        return 1.0, 0, []
    ratios = np.array(ratios)
    # robust center: median
    med = float(np.median(ratios))
    # count inliers within 2% of median
    tol = 0.02
    inliers = ratios[np.abs(ratios - med) <= tol * med]
    return med, len(inliers), ratios


def analyze(pair):
    path_a, path_b = pair
    freqs_a, spec_a, sr_a = mean_spectrum(path_a)
    freqs_b, spec_b, sr_b = mean_spectrum(path_b)
    peaks_a = top_peaks(freqs_a, spec_a, topk=40)
    peaks_b = top_peaks(freqs_b, spec_b, topk=40)

    med_ab, inliers_ab, ratios_ab = estimate_ratio(peaks_a, peaks_b)
    med_ba, inliers_ba, ratios_ba = estimate_ratio(peaks_b, peaks_a)

    # determine which has higher spectral frequencies
    # if med_ab > 1 -> peaks in A are higher than B -> A pitched up relative to B
    result = {
        'file_a': os.path.basename(path_a),
        'file_b': os.path.basename(path_b),
        'sr_a': sr_a,
        'sr_b': sr_b,
        'med_ab': med_ab,
        'inliers_ab': inliers_ab,
        'med_ba': med_ba,
        'inliers_ba': inliers_ba,
        'peaks_a': peaks_a.tolist(),
        'peaks_b': peaks_b.tolist(),
    }
    # pick direction with more inliers
    if inliers_ab >= inliers_ba:
        estimated_ratio = med_ab
        pitched_file = 'A' if med_ab > 1.0 else ('B' if med_ab < 1.0 else 'none')
    else:
        estimated_ratio = 1.0 / med_ba
        pitched_file = 'B' if med_ba > 1.0 else ('A' if med_ba < 1.0 else 'none')

    result['estimated_ratio'] = float(estimated_ratio)
    result['pitched_file'] = pitched_file
    return result


def main():
    files = sorted(glob.glob(os.path.join(TASK3_DIR, 'task3_watermarked_*.wav')))
    if len(files) < 2:
        # fallback: match any two wavs in folder
        files = sorted(glob.glob(os.path.join(TASK3_DIR, '*.wav')))
    if len(files) < 2:
        print('Task 3 files not found in', TASK3_DIR)
        return
    # assume exactly two target files
    f1, f2 = files[:2]
    print('Analyzing:')
    print('  1:', os.path.basename(f1))
    print('  2:', os.path.basename(f2))

    res = analyze((f1, f2))

    # Deduce methods: time-domain resampling causes pitch shift (peaks scaled by x),
    # frequency-domain time-scaling often preserves pitch (peaks match, ratio ~1)
    pitched = res['pitched_file']
    x = res['estimated_ratio']
    if pitched == 'A':
        time_domain_file = os.path.basename(f1)
        freq_domain_file = os.path.basename(f2)
    elif pitched == 'B':
        time_domain_file = os.path.basename(f2)
        freq_domain_file = os.path.basename(f1)
    else:
        # no clear pitch shift detected
        time_domain_file = None
        freq_domain_file = None

    lines = []
    lines.append('Task 3 Speedup analysis')
    lines.append('Files:')
    lines.append(' - A: ' + os.path.basename(f1))
    lines.append(' - B: ' + os.path.basename(f2))
    lines.append('')
    lines.append(f"Estimated spectral ratio (A/B): {res['med_ab']:.6f} with inliers={res['inliers_ab']}")
    lines.append(f"Estimated spectral ratio (B/A): {res['med_ba']:.6f} with inliers={res['inliers_ba']}")
    lines.append('')
    if time_domain_file:
        lines.append(f"Decision: time-domain resampling (pitch-shift) was applied to: {time_domain_file}")
        lines.append(f"Frequency-domain method was applied to: {freq_domain_file}")
        lines.append(f"Estimated speed-up ratio x ≈ {x:.4f} (spectral scaling) — i.e., frequencies scaled by x")
    else:
        lines.append('Decision: no clear pitch-scaling detected between the two files.')
        lines.append('Both files have similar spectral peak positions; frequency-domain method likely used for both, or x≈1.')

    # Add quick evidence: list a few matched peak pairs
    lines.append('')
    lines.append('Top peaks A (Hz): ' + ', '.join('%.1f' % p for p in res['peaks_a'][:12]))
    lines.append('Top peaks B (Hz): ' + ', '.join('%.1f' % p for p in res['peaks_b'][:12]))

    outp = os.path.join(OUT_DIR, 'task3_results.txt')
    with open(outp, 'w', encoding='utf8') as fh:
        fh.write('\n'.join(lines))

    print('\n'.join(lines))
    print('\nWrote results to', outp)

if __name__ == '__main__':
    main()
