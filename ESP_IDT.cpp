#include <Arduino.h>
const int mq2pin = D2;
void setup() {
    Serial.begin(115200);
    analogReadResolution(12);
    Serial.println("Sensors warming up...");
    delay(2000);
}
void loop() {
    int rawValue = analogRead(mq2pin);
    float percent = (rawValue / 4095.0)*100.0;
    Serial.printf("Gas level percentage = ", percent);
    if (percent>20) {
        Serial.println("ALERT: Gas leak detected.");
        delay(1000);
    }
}