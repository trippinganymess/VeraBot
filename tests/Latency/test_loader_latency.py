import json
import time
from pathlib import Path
import unittest

from bot import load_data

BASELINE_PATH = Path(__file__).parent / "baseline.json"
LOG_PATH = Path(__file__).parent.parent.parent / "logs" / "perf.log"


def _write_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


class TestLoaderLatency(unittest.TestCase):
    def test_loader_latency(self):
        samples = 3
        durations_ms = []
        for _ in range(samples):
            start = time.perf_counter()
            load_data()
            durations_ms.append((time.perf_counter() - start) * 1000)
        avg_ms = sum(durations_ms) / len(durations_ms)

        baseline = None
        if BASELINE_PATH.exists():
            baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

        if baseline is None:
            BASELINE_PATH.write_text(
                json.dumps({"avg_ms": avg_ms, "recorded_at": time.time()}),
                encoding="utf-8",
            )
            _write_log(f"baseline_created avg_ms={avg_ms:.2f}")
            return

        baseline_ms = baseline.get("avg_ms", avg_ms)
        delta_ms = avg_ms - baseline_ms
        _write_log(
            f"latency_check avg_ms={avg_ms:.2f} baseline_ms={baseline_ms:.2f} delta_ms={delta_ms:.2f}"
        )

        threshold_ms = baseline_ms * 1.10
        self.assertLessEqual(
            avg_ms,
            threshold_ms,
            msg=(
                f"Latency regression detected: avg {avg_ms:.2f}ms exceeds "
                f"baseline {baseline_ms:.2f}ms by >10%"
            ),
        )


if __name__ == "__main__":
    unittest.main()
