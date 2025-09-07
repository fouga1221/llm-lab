"""
Lightweight GPU monitor using NVML.

- Starts a background thread to sample GPU utilization (%) every `interval_sec`.
- Collects average utilization over the monitored window.
- Falls back gracefully if NVML is not available.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional


class NVMLMonitor:
    def __init__(self, device_index: int = 0, interval_sec: float = 0.1) -> None:
        self.device_index = device_index
        self.interval_sec = interval_sec
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._samples: List[float] = []
        self._ok = False
        self._err: Optional[str] = None

        try:
            import pynvml  # type: ignore

            self._pynvml = pynvml
            self._pynvml.nvmlInit()
            # will raise if invalid
            self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._ok = True
        except Exception as e:  # pragma: no cover - environment dependent
            self._err = str(e)
            self._ok = False
            self._pynvml = None
            self._handle = None

    def start(self) -> None:
        if not self._ok:
            return

        def _run() -> None:
            while not self._stop.is_set():
                try:
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)  # type: ignore[attr-defined]
                    self._samples.append(float(util.gpu))
                except Exception:
                    # swallow intermittent errors
                    pass
                time.sleep(self.interval_sec)

        self._stop.clear()
        self._samples.clear()
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def available(self) -> bool:
        return self._ok

    @property
    def error(self) -> Optional[str]:
        return self._err

    def mean_util(self) -> Optional[float]:
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)

    def samples(self) -> List[float]:
        return list(self._samples)

    def close(self) -> None:
        try:
            if self._pynvml is not None:
                self._pynvml.nvmlShutdown()  # type: ignore[attr-defined]
        except Exception:
            pass


__all__ = ["NVMLMonitor"]

