"""Métriques communes à tous les modèles, pour pouvoir les comparer sur le même format."""
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred, model, templates=None, **extra):
    """`templates` (facultatif) : gabarit de chaque exemple de test, pour l'accuracy par gabarit."""
    metrics = {
        "model": model,
        "n_test": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spam": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_spam": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_spam": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        **extra,
    }
    if templates is not None:
        per_template = {}
        for t, gold, pred in zip(templates, y_true, y_pred):
            ok, n = per_template.get(t, (0, 0))
            per_template[t] = (ok + int(gold == pred), n + 1)
        metrics["per_template"] = {t: {"correct": ok, "total": n} for t, (ok, n) in sorted(per_template.items())}
    return metrics
