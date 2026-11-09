import json
import os
import tempfile

from phase2_nlp_classique.benchmark_fr import run
from phase2_nlp_classique.metrics import compute_metrics
from scripts.compare_models import build_table


def test_compute_metrics_values():
    m = compute_metrics([1, 1, 0, 0], [1, 0, 0, 0], "demo")
    assert m["accuracy"] == 0.75 and m["precision_spam"] == 1.0 and m["recall_spam"] == 0.5
    assert m["confusion_matrix"] == [[2, 0], [1, 1]]


def test_tfidf_baseline_runs_on_unseen_templates():
    m = run(seed=0)
    assert 0.5 < m["accuracy"] <= 1.0 and m["n_test"] > 0


def test_build_table_reads_metric_files():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "demo_fr_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(compute_metrics([1, 0], [1, 0], "demo"), f)
        table = build_table(tmp)
    assert "| demo | 1.000 |" in table
    assert "référence : toujours spam | 0.500" in table


def test_per_template_accuracy():
    m = compute_metrics([1, 1, 0, 0], [1, 0, 0, 0], "demo", templates=["a", "a", "b", "b"])
    assert m["per_template"] == {"a": {"correct": 1, "total": 2}, "b": {"correct": 2, "total": 2}}
