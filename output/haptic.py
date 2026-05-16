"""
GPIO haptic motor driver using lgpio (Pi 5 native).
4 motors in X formation:
  GPIO  6 = front-left  (315°)  physical pin 31
  GPIO 16 = front-right ( 45°)  physical pin 36
  GPIO 26 = back-right  (135°)  physical pin 37
  GPIO 21 = back-left   (225°)  physical pin 40
(GPIO 18/19/20 reserved for I2S mic array; GPIO 12/13 for cooling fan)
Falls back to console print on non-Pi platforms.
"""
import math
import threading

# X-formation motor layout: (gpio_pin, angle_deg)
MOTORS = [
    ( 6, 315),  # front-left   pin 31
    (16,  45),  # front-right  pin 36
    (26, 135),  # back-right   pin 37
    (21, 225),  # back-left    pin 40
]
PWM_FREQ_HZ = 100

try:
    import lgpio
    _chip = lgpio.gpiochip_open(0)
    for _pin, _ in MOTORS:
        lgpio.gpio_claim_output(_chip, _pin)
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False
    _chip = None

URGENCY_DUTY = {
    "low": 0,
    "medium": 60,
    "high": 100,
    "critical": 100,
}

PATTERN_DURATION = {
    "none": 0.0,
    "slow_pulse": 1.2,
    "rapid_pulse": 0.8,
    "rapid_left": 0.6,
    "rapid_right": 0.6,
    "rapid_center": 0.6,
}

_MOTOR_LABELS = ["FL", "FR", "BL", "BR"]


class HapticController:
    def __init__(self):
        self._lock = threading.Lock()
        self._active_timer = None

    def fire(self, direction_deg: float, urgency: str, pattern: str = "none") -> None:
        duty = URGENCY_DUTY.get(urgency, 0)
        duration = PATTERN_DURATION.get(pattern, 0.6)
        if duty == 0:
            return

        duties = self._direction_to_duties(direction_deg, duty)

        with self._lock:
            if self._active_timer:
                self._active_timer.cancel()
            self._set_motors(duties)
            if duration > 0:
                self._active_timer = threading.Timer(duration, self._stop_motors)
                self._active_timer.daemon = True
                self._active_timer.start()

        if not GPIO_AVAILABLE:
            bar_parts = " ".join(f"{_MOTOR_LABELS[i]}={'█' * (d // 10):<10}" for i, d in enumerate(duties))
            print(f"  [HAPTIC   ] dir={direction_deg:5.1f}° {urgency:8s} {bar_parts} pat={pattern}")

    def _direction_to_duties(self, direction_deg: float, max_duty: int) -> list[int]:
        """Cosine activation over 4 X-formation motors.
        Each motor scales to max(0, cos(direction - motor_angle)).
        Normalised by peak weight so strongest motor always hits max_duty."""
        rad = math.radians(direction_deg)
        weights = []
        for _, angle in MOTORS:
            motor_rad = math.radians(angle)
            w = max(0.0, math.cos(rad - motor_rad))
            weights.append(w)
        peak = max(weights) or 1.0
        return [round(max_duty * w / peak) for w in weights]

    def _set_motors(self, duties: list[int]) -> None:
        if GPIO_AVAILABLE and _chip is not None:
            for (pin, _), duty in zip(MOTORS, duties):
                lgpio.tx_pwm(_chip, pin, PWM_FREQ_HZ, duty)

    def _stop_motors(self) -> None:
        if GPIO_AVAILABLE and _chip is not None:
            for pin, _ in MOTORS:
                lgpio.tx_pwm(_chip, pin, PWM_FREQ_HZ, 0)

    def cleanup(self) -> None:
        self._stop_motors()
        if GPIO_AVAILABLE and _chip is not None:
            lgpio.gpiochip_close(_chip)
