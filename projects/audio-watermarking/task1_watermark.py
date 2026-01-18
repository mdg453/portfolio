#!/usr/bin/env python3
"""
Simple audio watermark embedder and detector for Exercise 2 Task 1.

Good watermark: spread-spectrum pseudorandom noise bandpassed into a high-frequency
band (likely inaudible for average listeners) and scaled to be very low amplitude.

Bad watermark: a low-frequency audible tone added at higher amplitude (easy to hear).

Usage:
    python task1_watermark.py <input-wav>

Produces:
    outputs/good_watermarked.wav
    outputs/bad_watermarked.wav

Also prints correlation/detection scores for the good watermark.
"""

import os
import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt


def bandpass_filter(x, sr, low_hz, high_hz, order=6):
    nyq = 0.5 * sr
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999999)
    if low >= high:
        raise ValueError(f"Invalid bandpass ({low_hz},{high_hz}) for sr={sr}")
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x)


def embed_good(audio, sr, seed=12345, low_hz=14000, high_hz=20000, strength=0.002):
    # audio: (n_samples, n_channels)
    n_samples = audio.shape[0]
    rng = np.random.RandomState(seed)
    pn = rng.normal(size=n_samples)
    # bandpass PN into high-frequency range
    # cap high_hz to Nyquist
    max_high = max(0.49 * sr, high_hz)
    if high_hz >= sr/2:
        high = sr/2 - 100.0
    else:
        high = high_hz
    low = min(low_hz, high - 100.0)
    pn = bandpass_filter(pn, sr, low, high)
    # scale watermark relative to audio RMS
    audio_rms = np.sqrt(np.mean(audio**2) + 1e-16)
    pn_rms = np.sqrt(np.mean(pn**2) + 1e-16)
    alpha = strength * audio_rms / pn_rms
    wm = alpha * pn
    wm_stereo = np.tile(wm[:, None], (1, audio.shape[1]))
    return audio + wm_stereo, pn, (low, high, alpha)


def embed_bad(audio, sr, freq=1000.0, amplitude=0.02):
    n_samples = audio.shape[0]
    t = np.arange(n_samples) / float(sr)
    tone = amplitude * np.sin(2.0 * np.pi * freq * t)
    tone_stereo = np.tile(tone[:, None], (1, audio.shape[1]))
    return audio + tone_stereo, tone


def detect_good(watermarked_audio, sr, pn, band, alpha):
    # Filter the watermarked audio to the same band and correlate with PN
    low, high = band
    # combine channels by averaging
    mono = np.mean(watermarked_audio, axis=1)
    filtered = bandpass_filter(mono, sr, low, high)
    # correlate (normalized dot product)
    pn_norm = pn - np.mean(pn)
    filtered_norm = filtered - np.mean(filtered)
    corr = np.dot(filtered_norm, pn_norm) / (np.linalg.norm(filtered_norm) * np.linalg.norm(pn_norm) + 1e-16)
    return corr


def ensure_mono2d(data):
    if data.ndim == 1:
        data = data[:, None]
    return data


def main():
    if len(sys.argv) < 2:
        print("Usage: python task1_watermark.py <input-wav>")
        sys.exit(1)
    inp = sys.argv[1]
    outdir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(outdir, exist_ok=True)

    audio, sr = sf.read(inp, dtype='float32')
    audio = ensure_mono2d(audio)
    print(f"Loaded '{inp}' SR={sr} shape={audio.shape}")

    # Good watermark
    print("Embedding good (inaudible-like) watermark...")
    try:
        good_wm_audio, pn, params = embed_good(audio, sr)
        low, high, alpha = params
        out_good = os.path.join(outdir, 'good_watermarked.wav')
        sf.write(out_good, good_wm_audio, sr)
        print(f"Saved good watermark -> {out_good} (band {low:.0f}-{high:.0f} Hz, alpha={alpha:.6f})")
        corr = detect_good(good_wm_audio, sr, pn, (low, high), alpha)
        print(f"Detection correlation (good watermark): {corr:.6f}")
    except Exception as e:
        print("Failed to embed/detect good watermark:", e)

    # Bad watermark
    print("Embedding bad (audible) watermark...")
    bad_wm_audio, tone = embed_bad(audio, sr)
    out_bad = os.path.join(outdir, 'bad_watermarked.wav')
    sf.write(out_bad, bad_wm_audio, sr)
    print(f"Saved bad watermark -> {out_bad} (tone ~1kHz, amplitude approx {np.max(np.abs(tone)):.5f})")

    print("Done. Outputs are in the 'outputs' folder.")

if __name__ == '__main__':
    main()
