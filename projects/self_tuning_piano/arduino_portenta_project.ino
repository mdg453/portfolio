#define PICKUP_PIN A0
#define SOLENOID_IN1 4
#define SOLENOID_IN2 5

// הגדרות מנוע DC (JGY-370) באמצעות בקר סטנדרטי
const int MOTOR_IN1 = D8;
const int MOTOR_IN2 = D9;

// הגדרות דגימה (2048 הרץ)
const uint16_t NUM_SAMPLES = 2048;
const unsigned int SAMPLING_PERIOD_US = 488; // 1,000,000 us / 2048 Hz = ~488 us

void setup() {
  Serial.begin(250000); // קצב אגרסיבי כדי לא לתקוע את הבאפר
  analogReadResolution(12);

  pinMode(SOLENOID_IN1, OUTPUT);
  pinMode(SOLENOID_IN2, OUTPUT);
  digitalWrite(SOLENOID_IN1, LOW);
  digitalWrite(SOLENOID_IN2, LOW);

  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  digitalWrite(MOTOR_IN1, LOW);
  digitalWrite(MOTOR_IN2, LOW);

  pinMode(LEDB, OUTPUT);
  digitalWrite(LEDB, HIGH); // Active LOW: HIGH means OFF
}

void loop() {
  // ממתינים להוראה מהמחשב (פייתון)
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    // פקודה 'P' או 'p': פריטה ואיסוף נתונים
    if (cmd == 'P' || cmd == 'p') {
      digitalWrite(LEDB, LOW); // Turn ON Blue LED
      digitalWrite(SOLENOID_IN1, HIGH);
      delay(50);
      digitalWrite(SOLENOID_IN1, LOW);
      delay(1000); // זמן למיתר להתייצב - נותנים לו שנייה שלמה לעבור את שלב ה"התקפה" המרעיש

      unsigned long next_sample_time;
      // לולאת דגימה קריטית בזמן (Time-Critical)
      for (uint16_t i = 0; i < NUM_SAMPLES; i++) {
        next_sample_time = micros() + SAMPLING_PERIOD_US;
        int rawVal = analogRead(PICKUP_PIN);
        Serial.println(rawVal); // שליחת הדגימה לפייתון

        while (micros() < next_sample_time) { /* Active Wait */
        }
      }
      Serial.println("END");    // סימן לפייתון שהמערך נגמר
      digitalWrite(LEDB, HIGH); // Turn OFF Blue LED
    }

    // פקודה 'F' או 'f': מתיחה (הפעלת מנוע קדימה ל-500 מילישניות)
    else if (cmd == 'F' || cmd == 'f') {
      digitalWrite(MOTOR_IN1, HIGH);
      digitalWrite(MOTOR_IN2, LOW);
      delay(500);
      digitalWrite(MOTOR_IN1, LOW);
      Serial.println("DONE");
    }

    // פקודה 'B' או 'b': שחרור (הפעלת מנוע אחורה ל-500 מילישניות)
    else if (cmd == 'B' || cmd == 'b') {
      digitalWrite(MOTOR_IN1, LOW);
      digitalWrite(MOTOR_IN2, HIGH);
      delay(500);
      digitalWrite(MOTOR_IN2, LOW);
      Serial.println("DONE");
    }

    // פקודה 'S' או 's': הפעלת המנוע למשך זמן מוגדר במילישניות (למידת מכונה)
    else if (cmd == 'S' || cmd == 's') {
      long time_ms = Serial.parseInt();
      if (time_ms > 0) {
        digitalWrite(MOTOR_IN1, HIGH);
        digitalWrite(MOTOR_IN2, LOW);
        delay(time_ms);
        digitalWrite(MOTOR_IN1, LOW);
      } else if (time_ms < 0) {
        digitalWrite(MOTOR_IN1, LOW);
        digitalWrite(MOTOR_IN2, HIGH);
        delay(-time_ms);
        digitalWrite(MOTOR_IN2, LOW);
      }
      Serial.println("DONE");
    }
  }
}
