"""
Metrics calculation for evaluation.
Corresponds to section 10 of the detailed design document.
"""
from typing import List, Dict, Any, Iterable

from app.core.types import StructuredAction

class MetricsCalculator:
    """
    Calculates key performance indicators (KPIs) from evaluation logs.
    """

    def tool_accuracy(self, truth: List[StructuredAction], pred: List[StructuredAction]) -> float:
        """
        Calculates tool call accuracy (exact match).
        Compares function names and arguments.

        Args:
            truth: The ground truth list of actions.
            pred: The predicted list of actions.

        Returns:
            A score from 0.0 to 1.0. 1.0 means a perfect match.
        """
        if not truth and not pred:
            return 1.0
        if not truth or not pred:
            return 0.0

        # Simple exact match based on converting to a comparable format (e.g., sorted tuples)
        def _canonicalize(actions: List[StructuredAction]) -> set:
            s = set()
            for action in actions:
                # Convert arguments to a frozenset of items for hashability
                args = frozenset(action.get('arguments', {}).items())
                s.add((action.get('function'), args))
            return s

        truth_set = _canonicalize(truth)
        pred_set = _canonicalize(pred)

        return 1.0 if truth_set == pred_set else 0.0

    def json_valid_rate(self, logs: Iterable[Dict[str, Any]]) -> float:
        """
        Calculates the rate of valid JSON outputs from a collection of logs.

        Args:
            logs: An iterable of log records (as dictionaries).

        Returns:
            The percentage of logs where 'json_valid' is true.
        """
        total_logs = 0
        valid_count = 0
        for log in logs:
            if 'json_valid' in log:
                total_logs += 1
                if log['json_valid']:
                    valid_count += 1
        
        return (valid_count / total_logs) if total_logs > 0 else 0.0

    def latency_stats(self, values: List[float]) -> Dict[str, float]:
        """
        Calculates latency statistics (e.g., average, p50, p95, p99).

        Args:
            values: A list of latency measurements in milliseconds.

        Returns:
            A dictionary of statistics.
        """
        if not values:
            return {}
        
        import numpy as np
        
        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'p50': float(np.percentile(values, 50)),
            'p95': float(np.percentile(values, 95)),
            'p99': float(np.percentile(values, 99)),
            'max': float(np.max(values)),
        }
