Exercise 2 — Task 1

This small utility embeds two types of watermarks into the provided Task 1 audio (`ExIn/Task 1/task1.wav`):

- good_watermarked.wav: a spread-spectrum pseudorandom noise watermark bandpassed to a high-frequency band (aiming for inaudibility) and scaled to very low amplitude.
- bad_watermarked.wav: a simple audible low-frequency tone (1 kHz) inserted at higher amplitude (intentionally perceptible).

How to run (from the repository root):

```powershell
python .\task1_watermark.py .\ExIn\Task 1\task1.wav
```

Outputs are written to `outputs/` next to the script.

Notes:
- The script uses the input file's sampling rate; do not resample unless intended.
- "Good" watermark here is intentionally simple (spread-spectrum in HF). In a production system you'd use stronger/robust schemes and perceptual models.
- Listen to `outputs/good_watermarked.wav` and `outputs/bad_watermarked.wav` to judge inaudibility.
