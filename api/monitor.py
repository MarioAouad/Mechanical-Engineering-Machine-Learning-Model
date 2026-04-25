"""
Drift Monitoring Module — Acoustic Anomaly Detection (DCASE 2024 Task 2)

Tracks production inference statistics to detect:
  1. Feature drift   — Are spectrogram values shifting?
  2. Score drift     — Is the anomaly score distribution changing?
  3. Decision drift  — Is the system flagging too many / too few clips?

Usage:
  The monitor is integrated into the FastAPI server. Statistics are
  available via GET /health and GET /stats endpoints.

  Logs are written to api/monitoring_logs/ as JSONL files (one per day).
"""
import os
import json
import time
import threading
from datetime import datetime, timezone
from collections import defaultdict

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "predictions")
os.makedirs(LOGS_DIR, exist_ok=True)


class DriftMonitor:
    """
    Lightweight production monitoring for anomaly detection.

    Tracks per-machine rolling statistics:
      - Anomaly scores (mean, std, min, max)
      - Anomaly rate (fraction of clips flagged)
      - Spectrogram feature stats (mean pixel value)
      - Request counts and latencies
    """

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Number of recent predictions to keep in the
                         rolling window for drift detection.
        """
        self.window_size = window_size
        self.lock = threading.Lock()
        self.start_time = time.time()

        # Per-machine rolling windows
        self._scores = defaultdict(list)       # anomaly scores
        self._decisions = defaultdict(list)    # 0=normal, 1=anomaly
        self._latencies = defaultdict(list)    # inference time (seconds)
        self._spec_means = defaultdict(list)   # mean spectrogram pixel value
        self._total_requests = defaultdict(int)
        self._total_anomalies = defaultdict(int)

    def record(self, machine: str, score: float, is_anomaly: bool,
               latency: float, spec_mean: float):
        """Record one prediction for monitoring."""
        with self.lock:
            ws = self.window_size

            self._scores[machine].append(score)
            self._decisions[machine].append(1 if is_anomaly else 0)
            self._latencies[machine].append(latency)
            self._spec_means[machine].append(spec_mean)
            self._total_requests[machine] += 1
            if is_anomaly:
                self._total_anomalies[machine] += 1

            # Trim to window
            if len(self._scores[machine]) > ws:
                self._scores[machine] = self._scores[machine][-ws:]
                self._decisions[machine] = self._decisions[machine][-ws:]
                self._latencies[machine] = self._latencies[machine][-ws:]
                self._spec_means[machine] = self._spec_means[machine][-ws:]

        # Async log to file (non-blocking)
        self._log_to_file(machine, score, is_anomaly, latency, spec_mean)

    def _log_to_file(self, machine, score, is_anomaly, latency, spec_mean):
        """Append a JSONL log entry for this prediction."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_path = os.path.join(LOGS_DIR, f"predictions_{today}.jsonl")
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "machine": machine,
            "score": round(score, 8),
            "is_anomaly": is_anomaly,
            "latency_ms": round(latency * 1000, 1),
            "spec_mean": round(spec_mean, 6),
        }
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # Don't crash the server for a logging failure

    def get_stats(self) -> dict:
        """Return current monitoring statistics for all machines."""
        with self.lock:
            stats = {}
            for machine in sorted(self._scores.keys()):
                scores = self._scores[machine]
                decisions = self._decisions[machine]
                latencies = self._latencies[machine]
                spec_means = self._spec_means[machine]

                import numpy as np
                scores_arr = np.array(scores) if scores else np.array([0.0])
                decisions_arr = np.array(decisions) if decisions else np.array([0])
                latencies_arr = np.array(latencies) if latencies else np.array([0.0])
                spec_means_arr = np.array(spec_means) if spec_means else np.array([0.0])

                stats[machine] = {
                    "total_requests": self._total_requests[machine],
                    "total_anomalies": self._total_anomalies[machine],
                    "anomaly_rate_overall": (
                        self._total_anomalies[machine] / max(1, self._total_requests[machine])
                    ),
                    "window_size": len(scores),
                    "score_distribution": {
                        "mean": round(float(scores_arr.mean()), 6),
                        "std": round(float(scores_arr.std()), 6),
                        "min": round(float(scores_arr.min()), 6),
                        "max": round(float(scores_arr.max()), 6),
                    },
                    "anomaly_rate_window": round(float(decisions_arr.mean()), 4),
                    "latency_ms": {
                        "mean": round(float(latencies_arr.mean() * 1000), 1),
                        "p95": round(float(np.percentile(latencies_arr, 95) * 1000), 1),
                    },
                    "feature_drift": {
                        "spec_mean_current": round(float(spec_means_arr[-1]), 6),
                        "spec_mean_window_avg": round(float(spec_means_arr.mean()), 6),
                        "spec_mean_window_std": round(float(spec_means_arr.std()), 6),
                    },
                }

                # Drift alerts
                alerts = []
                if stats[machine]["anomaly_rate_window"] > 0.3:
                    alerts.append("HIGH_ANOMALY_RATE: >30% of recent clips flagged as anomalous")
                if stats[machine]["score_distribution"]["std"] < 1e-6:
                    alerts.append("COLLAPSED_SCORES: All scores identical — model may be dead")
                if abs(stats[machine]["feature_drift"]["spec_mean_current"] -
                       stats[machine]["feature_drift"]["spec_mean_window_avg"]) > \
                       3 * max(stats[machine]["feature_drift"]["spec_mean_window_std"], 1e-6):
                    alerts.append("FEATURE_DRIFT: Current spectrogram mean deviates >3σ from window")

                stats[machine]["alerts"] = alerts

            return stats

    def get_health(self) -> dict:
        """Return system health summary."""
        uptime = time.time() - self.start_time
        total_reqs = sum(self._total_requests.values())
        total_anomalies = sum(self._total_anomalies.values())
        stats = self.get_stats()
        all_alerts = []
        for m, s in stats.items():
            for alert in s.get("alerts", []):
                all_alerts.append(f"[{m}] {alert}")

        return {
            "status": "degraded" if all_alerts else "healthy",
            "uptime_seconds": round(uptime, 1),
            "total_requests": total_reqs,
            "total_anomalies": total_anomalies,
            "anomaly_rate": total_anomalies / max(1, total_reqs),
            "machines_served": list(sorted(self._total_requests.keys())),
            "active_alerts": all_alerts,
        }
