"""
Structured logging and metrics utilities.
"""
import time
import json
import logging
from typing import Any, Dict

class Logger:
    """
    A simple structured logger that outputs JSON Lines.
    Corresponds to section 4.2 of the detailed design document.
    """
    def __init__(self, name: str, level: str = 'INFO') -> None:
        """
        Initializes the logger.

        Args:
            name: The name of the logger.
            level: The logging level.
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.upper())
        # Avoid adding duplicate handlers if this class is instantiated multiple times
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            # We will format the log as JSON manually
            handler.setFormatter(logging.Formatter('%(message)s'))
            self._logger.addHandler(handler)

    def _log(self, level: str, msg: str, **fields: Any) -> None:
        log_record = {
            'level': level,
            'message': msg,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            **fields
        }
        self._logger.log(logging.getLevelName(level.upper()), json.dumps(log_record, ensure_ascii=False))

    def info(self, msg: str, **fields: Any) -> None:
        """Logs a message with INFO level."""
        self._log('info', msg, **fields)

    def error(self, msg: str, **fields: Any) -> None:
        """Logs a message with ERROR level."""
        self._log('error', msg, **fields)

    def warning(self, msg: str, **fields: Any) -> None:
        """Logs a message with WARNING level."""
        self._log('warning', msg, **fields)

class Metrics:
    """
    A simple metrics collector for timing and counting events.
    Corresponds to section 4.2 of the detailed design document.
    """
    def __init__(self) -> None:
        self._timers: Dict[str, float] = {}
        self._counters: Dict[str, float] = {}

    def start_timer(self, key: str) -> None:
        """
        Starts a timer for a given key.

        Args:
            key: The identifier for the timer.
        """
        self._timers[key] = time.perf_counter()

    def stop_timer(self, key: str) -> float:
        """
        Stops a timer and returns the elapsed time in milliseconds.

        Args:
            key: The identifier for the timer.

        Returns:
            The elapsed time in milliseconds, or -1.0 if the timer was not started.
        """
        if key in self._timers:
            elapsed_s = time.perf_counter() - self._timers.pop(key)
            return elapsed_s * 1000
        return -1.0

    def incr(self, key: str, value: float = 1.0) -> None:
        """
        Increments a counter by a given value.

        Args:
            key: The identifier for the counter.
            value: The value to increment by.
        """
        self._counters.setdefault(key, 0.0)
        self._counters[key] += value

    def get_counters(self) -> Dict[str, float]:
        """Returns the current state of all counters."""
        return self._counters.copy()
