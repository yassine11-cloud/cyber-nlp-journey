"""Rassemble les métriques des modèles dans un tableau Markdown à coller dans le README.

Ajoute deux références triviales (répondre toujours « spam », ou toujours « ham ») : un modèle utile doit
faire nettement mieux que ces références.

Lancer après avoir exécuté les modèles :
    python scripts/compare_models.py
"""
import glob
import json
import os
import sys


def _baselines(confusion_matrix):
    (tn, fp), (fn, tp) = confusion_matrix
    ham, spam = tn + fp, fn + tp
    n = ham + spam
    p = spam / n
    f1_spam = 2 * p / (1 + p)
    return [
        ("référence : toujours spam", p, p, 1.0, f1_spam),
        ("référence : toujours ham", ham / n, 0.0, 0.0, 0.0),
    ]


def build_table(directory="reports"):
    rows, first = [], None
    for path in sorted(glob.glob(os.path.join(directory, "*_fr_metrics.json"))):
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        first = first or m["confusion_matrix"]
        rows.append(f"| {m['model']} | {m['accuracy']:.3f} | {m['precision_spam']:.3f} | "
                    f"{m['recall_spam']:.3f} | {m['f1_spam']:.3f} |")
    if not rows:
        return "Aucun fichier *_fr_metrics.json trouvé."
    for name, acc, prec, rec, f1 in _baselines(first):
        rows.append(f"| {name} | {acc:.3f} | {prec:.3f} | {rec:.3f} | {f1:.3f} |")
    header = "| Modèle | Accuracy | Précision (spam) | Rappel (spam) | F1 (spam) |\n|---|---|---|---|---|"
    return header + "\n" + "\n".join(rows)


if __name__ == "__main__":
    print(build_table(sys.argv[1] if len(sys.argv) > 1 else "reports"))
