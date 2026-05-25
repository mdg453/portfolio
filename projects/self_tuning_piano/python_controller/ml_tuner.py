import os
import pickle
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

MODEL_FILE = 'tuner_model.pkl'

class MLTuner:
    def __init__(self):
        self.model = SGDRegressor(learning_rate='constant', eta0=0.01)
        self.scaler = StandardScaler()
        
        if os.path.exists(MODEL_FILE):
            print("Loading existing ML model...")
            with open(MODEL_FILE, 'rb') as f:
                saved_data = pickle.load(f)
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.is_initialized = True
        else:
            print("Initializing new ML model...")
            self.is_initialized = False
            # אימון ראשוני פשוט - נניח שיחס ראשוני הוא 100 מילישניות לכל 1 הרץ שינוי
            # Features: [current_freq, expected_delta_freq]
            # Target: time_ms
            initial_X = []
            initial_y = []
            for f in [100.0, 200.0, 300.0, 400.0]:
                for delta in [-10.0, -5.0, 5.0, 10.0]:
                    initial_X.append([f, delta])
                    initial_y.append(delta * 100.0) # 100 ms per Hz initial guess
            
            initial_X = np.array(initial_X)
            self.scaler.fit(initial_X)
            X_scaled = self.scaler.transform(initial_X)
            self.model.partial_fit(X_scaled, initial_y)
            self.is_initialized = True
            self.save_model()

    def predict_time_ms(self, current_freq, target_freq):
        diff = target_freq - current_freq
        X = np.array([[current_freq, diff]])
        X_scaled = self.scaler.transform(X)
        predicted_time = self.model.predict(X_scaled)[0]
        
        # הגבלת זמן הפעלה למניעת קריעת מיתר (2 שניות מקסימום למכה - מנוע JGY-370 חזק מאוד)
        max_time_ms = 2000
        if predicted_time > max_time_ms:
            predicted_time = max_time_ms
        elif predicted_time < -max_time_ms:
            predicted_time = -max_time_ms
            
        # מניעת פולסים קצרים מדי. הקטנו ל-50 מילישניות כדי לאפשר כיוון עדין יותר.
        min_time_ms = 50
        if 0 < predicted_time < min_time_ms:
            predicted_time = min_time_ms
        elif -min_time_ms < predicted_time < 0:
            predicted_time = -min_time_ms
            
        # --- Physics Constraint ---
        # חוקי הפיזיקה: חייבים להתקדם לכיוון הנכון.
        # אם המודל טעה בכיוון בגלל שהוא שמרן, ניתן לו קפיצה אמיצה יחסית למרחק (כ-20 מ"ש לכל הרץ)
        delta = target_freq - current_freq
        override_triggered = False
        
        if delta > 0 and predicted_time < 0:
            print(" [Physics Override] Model suggested loosening, but we need to go UP! Overriding to proportional +")
            predicted_time = max(min_time_ms, int(delta * 20.0))
            override_triggered = True
        elif delta < 0 and predicted_time > 0:
            print(" [Physics Override] Model suggested tightening, but we need to go DOWN! Overriding to proportional -")
            predicted_time = min(-min_time_ms, int(delta * 20.0))
            override_triggered = True
            
        if override_triggered:
            # למידה מכוונת (Directed Learning): נעניש את המודל על הטעות ונכריח אותו ללמוד את הכיוון הנכון באופן מיידי!
            synthetic_X = self.scaler.transform(np.array([[current_freq, delta]]))
            # משקל ענק (פי 100) כדי "לשבור" את ההטיה השגויה של המודל מיד
            self.model.partial_fit(synthetic_X, [predicted_time], sample_weight=[100.0])
            self.save_model()
            print(" [Directed Learning] Explicitly trained model with correct physics sign!")
            
        return int(predicted_time)

    def learn_from_step(self, start_freq, end_freq, time_ms_taken):
        delta_freq = end_freq - start_freq
        
        # התעלמות מקפיצות מטורפות (אוקטבות/הרמוניות)
        if abs(delta_freq) > 30.0 and abs(time_ms_taken) < 500:
            print(f" [ML Ignore] Detected harmonic jump ({delta_freq:.2f} Hz). Ignoring data.")
            return True
            
        # חוקי הפיזיקה - מתיחה מגדילה תדר, שחרור מוריד תדר.
        # אם המנוע מתח אבל התדר ירד משמעותית (מעל 2 הרץ) - המיתר החליק או שהזיהוי שגוי.
        if (time_ms_taken > 0 and delta_freq < -2.0) or (time_ms_taken < 0 and delta_freq > 2.0):
            print(f" [ML Ignore] Physics violation: Motor {time_ms_taken}ms -> {delta_freq:.2f}Hz. String slipped/Bad read.")
            return True # מתעלמים ולא מלמדים את המודל פיזיקה הפוכה!
            
        # זיהוי כשל מנוע / החלקה: המנוע עבד הרבה זמן אבל התדר כמעט לא השתנה
        if abs(time_ms_taken) >= 1000 and abs(delta_freq) < 1.0:
            return False # כשל
            
        if abs(delta_freq) < 0.1:
            return True # שינוי קטן מדי ללמידה
            
        X = np.array([[start_freq, delta_freq]])
        # עדכון הסקיילר - טכנית עדיף לא לעדכן Online אלא אם משתמשים ב-PartialFitScaler, 
        # אבל לצרכים שלנו נשאיר את הסקיילר קבוע כדי לשמור על יציבות
        X_scaled = self.scaler.transform(X)
        
        # פונקציית משקל: תנועות גדולות מקבלות חיזוק משמעותי יותר כדי לא להיתקע בצעדים קטנים
        weight = max(1.0, abs(delta_freq))
        
        self.model.partial_fit(X_scaled, [time_ms_taken], sample_weight=[weight])
        self.save_model()
        return True

    def save_model(self):
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
