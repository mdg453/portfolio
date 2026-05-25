import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from ml_tuner import MLTuner

# הגדרות תקשורת

SERIAL_PORT = '/dev/tty.usbmodem101'  # שנה בהתאם לפורט במק (למשל /dev/tty.usbmodem... או /dev/tty.usbserial...)
BAUD_RATE = 250000
FS = 2048.0  # תדר דגימה זהה למה שהוגדר בארדואינו

def connect_arduino():
    print(f"Connecting to {SERIAL_PORT}...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)  # חובה לתת לארדואינו זמן לבצע Reset אחרי חיבור
    ser.reset_input_buffer()
    print("Connected.")
    return ser

def get_string_frequency(ser):
    print("Commanding PLUCK...")
    ser.write(b'P')  # שליחת פקודת הפריטה
    
    raw_data = []
    print("Receiving data stream...")
    
    # קריאת הנתונים עד שמקבלים את מילת הקוד "END"
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line == "END":
                break
            if line.isdigit():
                raw_data.append(int(line))
        except Exception as e:
            pass # התעלמות משגיאות קידוד קטנות שאולי קורות בתחילת תקשורת

    if len(raw_data) == 0:
        print("Error: No data received.")
        return None

    print(f"Received {len(raw_data)} samples. Processing FFT...")
    
    # --- תחילת בלוק עיבוד אותות (DSP) באמצעות NumPy ו-SciPy ---
    
    signal = np.array(raw_data, dtype=float)
    
    # 0. שער רעש (Noise Gate) - בדיקה אם יש בכלל פריטה
    rms = np.sqrt(np.mean(np.square(signal - np.mean(signal))))
    if rms < 5.0: # סף עוצמה נמוך מאוד - כנראה שאין פריטה או שהכל רעש רקע
        print(f"Signal too weak (RMS={rms:.2f}). Please pluck louder.")
        return None
        
    # 1. סינון תדרים גבוהים (Low-Pass Filter) לחיתוך רעשי שריגים וחשמל מעל 600 הרץ
    nyq = 0.5 * FS
    cutoff = 600.0 / nyq
    b, a = butter(4, cutoff, btype='low', analog=False)
    signal = filtfilt(b, a, signal)
    
    # 2. הסרת ה-DC Bias (מרכוז הגל)
    signal = signal - np.mean(signal)
    
    # 3. חלון (Windowing) למניעת דליפה ספקטרלית
    window = np.hamming(len(signal))
    signal = signal * window
    
    # 4. התמרת פורייה
    fft_result = np.fft.rfft(signal)
    fft_magnitude = np.abs(fft_result)
    
    # 4. מציאת התדר בעזרת Harmonic Product Spectrum (HPS)
    # בגיטרות בס תדר היסוד חלש בהרבה מההרמוניות שלו.
    # אלגוריתם HPS (מכפלת ספקטרומים) מוצא את התדר שההרמוניות שלו חופפות הכי טוב.
    frequencies = np.fft.rfftfreq(len(signal), d=1.0/FS)
    
    # נשתמש בסכום לוגריתמים (שקול למכפלה) כדי לא לחרוג מגבולות מספריים
    log_mag = np.log1p(fft_magnitude)
    hps_log = np.copy(log_mag)
    
    num_harmonics = 4 # נבדוק התאמה לעד 4 הרמוניות
    for h in range(2, num_harmonics + 1):
        decimated = log_mag[::h]
        hps_log[:len(decimated)] += decimated
        
    # איפוס תדרים לא הגיוניים לגיטרה בס (מתחת ל-30 הרץ) ורעשי DC
    min_freq_idx = int(30.0 / (FS / len(signal)))
    hps_log[:min_freq_idx] = 0
    
    # הפיק הכי חזק בגרף ה-HPS הוא תדר היסוד האמיתי!
    peak_index = np.argmax(hps_log)
    fundamental_freq = frequencies[peak_index]
    
    # --- הצגת הגרפים ---
    plt.figure(1, figsize=(10, 6))
    plt.clf() # ניקוי הגרף הקודם
    
    plt.subplot(2, 1, 1)
    time_axis = np.arange(len(raw_data)) / FS
    plt.plot(time_axis, raw_data, color='blue')
    plt.title("Raw Audio Signal (Time Domain)")
    plt.xlabel("Time [s]")
    plt.ylabel("ADC Value")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(frequencies, fft_magnitude, color='red')
    plt.title("Frequency Spectrum (FFT)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.xlim(0, 1000) # נציג עד 1000 הרץ
    plt.axvline(x=fundamental_freq, color='green', linestyle='--', label=f'Peak: {fundamental_freq:.2f} Hz')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    
    return fundamental_freq

def auto_tune(ser, target_freq, ml_tuner, tolerance=0.9, max_iterations=20):
    print(f"\n--- ML Auto Tuning to {target_freq} Hz ---")
    
    last_freq = None
    last_time_ms = 0
    
    for i in range(max_iterations):
        freq = get_string_frequency(ser)
        if freq is None:
            print("Failed to measure frequency. Retrying...")
            continue
            
        if last_freq is not None and last_time_ms != 0:
            success = ml_tuner.learn_from_step(last_freq, freq, last_time_ms)
            if not success:
                print(f"\n[!] ERROR: Motor slip detected! Commanded {last_time_ms} ms, but frequency only changed by {(freq - last_freq):.2f} Hz.")
                print("Stopping auto-tune to prevent damage and avoid corrupting the ML model.")
                return False
            print(f"ML Model learned: {last_time_ms} ms changed freq by {(freq - last_freq):.2f} Hz.")
            
        print(f"Current: {freq:.2f} Hz | Target: {target_freq} Hz | Diff: {abs(freq - target_freq):.2f} Hz")
        
        if abs(freq - target_freq) <= tolerance:
            print(f"*** Success! Tuned to {freq:.2f} Hz ***\n")
            return True
            
        predicted_time_ms = ml_tuner.predict_time_ms(freq, target_freq)
        
        if predicted_time_ms == 0:
            predicted_time_ms = 100 if freq < target_freq else -100
            
        print(f"ML Model suggests {predicted_time_ms} ms.")
        
        ser.write(f'S{predicted_time_ms}\n'.encode('utf-8'))
        response = ser.readline().decode('utf-8').strip()
        print(f"Arduino response: {response}")
        
        last_freq = freq
        last_time_ms = predicted_time_ms
        
        time.sleep(1)
    
    print("Auto-tune stopped: Max iterations reached without perfect tune.\n")
    return False

def interactive_menu(ser, ml_tuner):
    while True:
        print("\n=== Guitar Tuner Menu ===")
        print("P - Pluck string and measure frequency")
        print("F[ms] - Tighten string (e.g. F 1000 for 1 sec)")
        print("B[ms] - Loosen string (e.g. B 1000 for 1 sec)")
        print("H<freq> - Auto tune to specific frequency (e.g. H220)")
        print("T - Automated ML Training (30 random iterations 100-440Hz)")
        print("Q - Quit")
        
        choice = input("Enter command: ").strip().upper()
        
        if choice == 'P':
            freq = get_string_frequency(ser)
            if freq is not None:
                print(f"=====================================")
                print(f">>> Detected Frequency: {freq:.2f} Hz <<<")
                print(f"=====================================")
        elif choice.startswith('F'):
            ms = 500
            val_str = choice[1:].strip()
            if val_str:
                try: ms = int(val_str)
                except ValueError: pass
            print(f"Tightening string (Motor Forward for {ms} ms)...")
            ser.write(f'S{ms}\n'.encode('utf-8'))
            response = ser.readline().decode('utf-8').strip()
            print(f"Arduino response: {response}")
        elif choice.startswith('B'):
            ms = 500
            val_str = choice[1:].strip()
            if val_str:
                try: ms = int(val_str)
                except ValueError: pass
            print(f"Loosening string (Motor Backward for {ms} ms)...")
            ser.write(f'S{-ms}\n'.encode('utf-8'))
            response = ser.readline().decode('utf-8').strip()
            print(f"Arduino response: {response}")
        elif choice == 'T':
            import random
            print("Starting automated ML training (30 iterations)...")
            for i in range(30):
                target = float(random.randint(50, 90))
                print(f"\n[TRAINING {i+1}/30] Tuning to {target:.2f} Hz...")
                auto_tune(ser, target, ml_tuner, tolerance=0.2, max_iterations=15)
                time.sleep(2)
            print("Training session complete!")
        elif choice.startswith('H'):
            try:
                target_freq = float(choice[1:])
                auto_tune(ser, target_freq, ml_tuner)
            except ValueError:
                print("Invalid target frequency. Use format H220")
        elif choice == 'Q':
            print("Exiting...")
            break
        else:
            print("Invalid command. Please try again.")

if __name__ == '__main__':
    arduino = connect_arduino()
    ml_tuner = MLTuner()
    
    try:
        interactive_menu(arduino, ml_tuner)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        arduino.close()