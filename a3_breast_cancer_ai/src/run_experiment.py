from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT.parent / ".packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


RANDOM_STATE = 42


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def evaluate_model(name: str, model, x_train, x_test, y_train, y_test) -> dict:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    else:
        y_score = model.decision_function(x_test)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_malignant": precision_score(y_test, y_pred),
        "recall_malignant": recall_score(y_test, y_pred),
        "specificity_benign": specificity_score(y_test, y_pred),
        "f1_malignant": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def make_models() -> dict:
    return {
        "Interpretable decision tree": DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=12,
            random_state=RANDOM_STATE,
        ),
        "Gaussian naive Bayes": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", GaussianNB()),
            ]
        ),
        "Logistic regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Multi-layer perceptron": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(24, 12),
                        activation="relu",
                        alpha=0.01,
                        early_stopping=True,
                        max_iter=1500,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
    headers = list(formatted.columns)
    rows = formatted.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def save_dataset_summary(data, x: pd.DataFrame, y: pd.Series, out_dir: Path) -> None:
    target_counts = y.map({0: "benign", 1: "malignant"}).value_counts().rename_axis("class")
    summary = {
        "instances": int(x.shape[0]),
        "features": int(x.shape[1]),
        "class_counts": target_counts.to_dict(),
        "feature_names": list(data.feature_names),
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_figures(models: dict, fitted_models: dict, x_test, y_test, feature_names, out_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    for name, model in fitted_models.items():
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(x_test)[:, 1]
        else:
            scores = model.decision_function(x_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#777", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves on held-out test set")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png", dpi=180)
    plt.close()

    best = fitted_models["Random forest"]
    importances = pd.Series(best.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances.values, y=importances.index, color="#31688e")
    plt.xlabel("Mean decrease in impurity")
    plt.ylabel("Feature")
    plt.title("Top random forest feature importances")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=180)
    plt.close()

    tree = fitted_models["Interpretable decision tree"]
    plt.figure(figsize=(14, 7))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=["benign", "malignant"],
        filled=True,
        rounded=True,
        fontsize=7,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "decision_tree_rules.png", dpi=180)
    plt.close()

    y_pred = best.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["benign", "malignant"],
        yticklabels=["benign", "malignant"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Random forest confusion matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_rf.png", dpi=180)
    plt.close()


def save_cross_validation(models: dict, x: pd.DataFrame, y: pd.Series, out_dir: Path) -> None:
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, x, y, scoring=scoring, cv=cv, n_jobs=1)
        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "cross_validation_metrics.csv", index=False)


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_breast_cancer()
    x = pd.DataFrame(data.data, columns=data.feature_names)
    # In scikit-learn this dataset encodes malignant as 0 and benign as 1.
    # The project flips it so 1 means malignant, matching clinical risk wording.
    y = pd.Series(1 - data.target, name="malignant")

    save_dataset_summary(data, x, y, out_dir)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = make_models()
    fitted_models = {}
    rows = []
    for name, model in models.items():
        rows.append(evaluate_model(name, model, x_train, x_test, y_train, y_test))
        fitted_models[name] = model

    metrics = pd.DataFrame(rows).drop(columns=["confusion_matrix"])
    metrics.to_csv(out_dir / "test_metrics.csv", index=False)
    (out_dir / "test_metrics.md").write_text(dataframe_to_markdown(metrics), encoding="utf-8")

    confusion = {row["model"]: row["confusion_matrix"] for row in rows}
    (out_dir / "confusion_matrices.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")

    tree = fitted_models["Interpretable decision tree"]
    rules = export_text(tree, feature_names=list(data.feature_names))
    (out_dir / "interpretable_tree_rules.txt").write_text(rules, encoding="utf-8")

    save_cross_validation(models, x, y, out_dir)
    save_figures(models, fitted_models, x_test, y_test, list(data.feature_names), out_dir)

    print("Experiment completed. Outputs written to:", out_dir)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
