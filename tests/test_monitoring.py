import tempfile
from pathlib import Path
import unittest

from scripts.watch_training import assess, finite_report, metrics_from_log


def metric(epoch, accuracy, loss=0.8, lr=0.1):
    return {"epoch": epoch, "learning_rate": lr,
            "train": {"accuracy": 0.85, "loss": loss},
            "validation": {"accuracy": accuracy, "loss": loss}}


class MonitoringTests(unittest.TestCase):
    def test_isolated_development_drop_does_not_stop_training(self):
        records = [metric(11, 0.7268), metric(12, 0.8112), metric(13, 0.8024), metric(14, 0.7364)]
        self.assertEqual(assess(records)["decision"], "continue")

    def test_nonfinite_and_sustained_failure_are_detected(self):
        self.assertEqual(assess([metric(12, 0.8, loss=float("nan"))])["decision"], "stop_recommended")
        self.assertEqual(assess([metric(i, 0.5) for i in range(41, 51)])["decision"], "stop_recommended")

    def test_log_replay_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "log"
            import json
            line = json.dumps(metric(12, 0.81))
            path.write_text("Debugger warning\n" + line + "\n" + line + "\n")
            self.assertEqual(len(metrics_from_log(path)), 1)

    def test_failure_report_can_record_nonfinite_metrics(self):
        import json
        report = {"health": assess([metric(12, 0.8, loss=float("nan"))]), "loss": float("nan")}
        restored = json.loads(json.dumps(finite_report(report), allow_nan=False))
        self.assertEqual(restored["health"]["decision"], "stop_recommended")
        self.assertIsNone(restored["loss"])


if __name__ == "__main__":
    unittest.main()
