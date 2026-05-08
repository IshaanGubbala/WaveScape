"""
GPIO haptic motor driver using lgpio (Pi 5 native).
Left temple: GPIO 24  Right temple: GPIO 25
(GPIO 18/19/20 reserved for I2S mic array; GPIO 12/13 for cooling fan)
Falls back to console print on non-Pi platforms.
"""
import time
import threading

GPIO_LEFT = 24
GPIO_RIGHT = 25
PWM_FREQ_HZ = 200

try:
    import lgpio
    _chip = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_output(_chip, GPIO_LEFT)
    lgpio.gpio_claim_output(_chip, GPIO_RIGHT)
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False
    _chip = None

URGENCY_DUTY = {
    "low": 0,
    "medium": 40,
    "high": 75,
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


class HapticController:
    def __init__(self):
        self._lock = threading.Lock()
        self._active_timer = None

    def fire(self, direction_deg: float, urgency: str, pattern: str = "none") -> None:
        duty = URGENCY_DUTY.get(urgency, 0)
        duration = PATTERN_DURATION.get(pattern, 0.6)
        if duty == 0:
            return

        right_bias = self._direction_to_split(direction_deg)
        left_duty = round(duty * (1.0 - right_bias))
        right_duty = round(duty * right_bias)

        with self._lock:
            if self._active_timer:
                self._active_timer.cancel()
            self._set_motors(left_duty, right_duty)
            if duration > 0:
                self._active_timer = threading.Timer(duration, self._stop_motors)
                self._active_timer.daemon = True
                self._active_timer.start()

        if not GPIO_AVAILABLE:
            side = "LEFT " if direction_deg > 180 else "RIGHT"
            bar = "█" * (duty // 10)
            print(f"  [HAPTIC   ] {side} dir={direction_deg:5.1f}° {urgency:8s} {bar} pat={pattern}")

    def _direction_to_split(self, deg: float) -> float:
        """Right-motor fraction 0.0–1.0. Smooth piecewise-linear around compass:
        0°=0.5 (50/50), 90°=1.0 (all R), 180°=0.5 (50/50), 270°=0.0 (all L), 360°=0.5.
        Continuous across 270→360 boundary."""
        deg = deg % 360
        if deg <= 90:
            return 0.5 + (deg / 90) * 0.5
        elif deg <= 180:
            return 1.0 - ((deg - 90) / 90) * 0.5
        elif deg <= 270:
            return 0.5 - ((deg - 180) / 90) * 0.5
        else:
            return ((deg - 270) / 90) * 0.5

    def _set_motors(self, left_duty: int, right_duty: int) -> None:
        if GPIO_AVAILABLE and _chip is not None:
            lgpio.tx_pwm(_chip, GPIO_LEFT, PWM_FREQ_HZ, left_duty)
            lgpio.tx_pwm(_chip, GPIO_RIGHT, PWM_FREQ_HZ, right_duty)

    def _stop_motors(self) -> None:
        if GPIO_AVAILABLE and _chip is not None:
            lgpio.tx_pwm(_chip, GPIO_LEFT, 0, 0)
            lgpio.tx_pwm(_chip, GPIO_RIGHT, 0, 0)

    def cleanup(self) -> None:
        self._stop_motors()
        if GPIO_AVAILABLE and _chip is not None:
            lgpio.gpiochip_close(_chip)
