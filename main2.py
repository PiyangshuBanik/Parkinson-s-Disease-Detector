import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split


DATASET_PATH = "parkinsons.csv"
TARGET_ACCURACY = 0.93


def load_and_train_model():
    parkinsons_dataset = pd.read_csv(DATASET_PATH)

    x = parkinsons_dataset.drop(columns=["name", "status"], axis=1)
    y = parkinsons_dataset["status"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Random Forest does not require feature scaling.
    base_model = RandomForestClassifier(random_state=42, class_weight="balanced")

    parameter_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 8, 12, 16],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=parameter_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)

    model = grid_search.best_estimator_
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, train_predictions),
        "test_accuracy": accuracy_score(y_test, test_predictions),
        "precision": precision_score(y_test, test_predictions, zero_division=0),
        "recall": recall_score(y_test, test_predictions, zero_division=0),
        "f1_score": f1_score(y_test, test_predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, test_predictions),
        "classification_report": classification_report(y_test, test_predictions, zero_division=0),
        "best_parameters": grid_search.best_params_,
        "cross_validation_accuracy": cross_val_score(model, x, y, cv=cv, scoring="accuracy").mean(),
        "feature_names": list(x.columns),
    }

    print_model_performance(metrics)
    return model, metrics


def print_model_performance(metrics):
    print("\nRandom Forest Model Performance")
    print("-" * 40)
    print(f"Training Accuracy: {metrics['train_accuracy'] * 100:.2f}%")
    print(f"Testing Accuracy: {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Cross-validation Accuracy: {metrics['cross_validation_accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall: {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score: {metrics['f1_score'] * 100:.2f}%")
    print(f"Best Parameters: {metrics['best_parameters']}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
    print("\nClassification Report:")
    print(metrics["classification_report"])

    if metrics["test_accuracy"] >= TARGET_ACCURACY:
        print("Target reached: model test accuracy is at least 93%.")
    else:
        print("Target not reached: try adding more data or adjusting model parameters.")


def create_app(model, metrics):
    root = tk.Tk()
    root.title("Parkinson's Disease Prediction")
    root.geometry("880x720")
    root.minsize(760, 620)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background="#f4f7fb")
    style.configure("Header.TLabel", background="#f4f7fb", foreground="#14324a", font=("Segoe UI", 18, "bold"))
    style.configure("Subheader.TLabel", background="#f4f7fb", foreground="#526879", font=("Segoe UI", 10))
    style.configure("TLabel", background="#f4f7fb", foreground="#253746", font=("Segoe UI", 10))
    style.configure("Metric.TLabel", background="#ffffff", foreground="#14324a", font=("Segoe UI", 10, "bold"))
    style.configure("TButton", font=("Segoe UI", 10), padding=8)
    style.configure("Predict.TButton", background="#1f7a5c", foreground="#ffffff")

    entries = {}

    main_frame = ttk.Frame(root, padding=18)
    main_frame.pack(fill="both", expand=True)

    ttk.Label(main_frame, text="Parkinson's Disease Prediction", style="Header.TLabel").pack(anchor="w")
    ttk.Label(
        main_frame,
        text="Enter the voice measurement values below and run the trained Random Forest model.",
        style="Subheader.TLabel",
    ).pack(anchor="w", pady=(4, 14))

    metrics_frame = ttk.Frame(main_frame)
    metrics_frame.pack(fill="x", pady=(0, 12))

    metric_items = [
        ("Training Accuracy", metrics["train_accuracy"]),
        ("Testing Accuracy", metrics["test_accuracy"]),
        ("CV Accuracy", metrics["cross_validation_accuracy"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1 Score", metrics["f1_score"]),
    ]

    for column, (label, value) in enumerate(metric_items):
        card = tk.Frame(metrics_frame, bg="#ffffff", highlightbackground="#d7e0ea", highlightthickness=1)
        card.grid(row=0, column=column, padx=4, sticky="nsew")
        metrics_frame.columnconfigure(column, weight=1)
        tk.Label(card, text=label, bg="#ffffff", fg="#526879", font=("Segoe UI", 9)).pack(pady=(8, 2))
        tk.Label(card, text=f"{value * 100:.2f}%", bg="#ffffff", fg="#14324a", font=("Segoe UI", 12, "bold")).pack(
            pady=(0, 8)
        )

    content_frame = ttk.Frame(main_frame)
    content_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(content_frame, bg="#f4f7fb", highlightthickness=0)
    scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
    form_frame = ttk.Frame(canvas)

    form_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=form_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    feature_names = metrics["feature_names"]
    for index, feature_name in enumerate(feature_names):
        row = index // 2
        column = (index % 2) * 2

        ttk.Label(form_frame, text=feature_name).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=6)
        entry = ttk.Entry(form_frame, width=22)
        entry.grid(row=row, column=column + 1, sticky="ew", padx=(0, 18), pady=6)
        entries[feature_name] = entry

    for column in range(4):
        form_frame.columnconfigure(column, weight=1)

    result_frame = tk.Frame(main_frame, bg="#ffffff", highlightbackground="#d7e0ea", highlightthickness=1)
    result_frame.pack(fill="x", pady=(12, 0))

    result_label = tk.Label(
        result_frame,
        text="Prediction result will appear here.",
        bg="#ffffff",
        fg="#14324a",
        font=("Segoe UI", 12, "bold"),
        pady=12,
    )
    result_label.pack(fill="x")

    def reset_values():
        for entry in entries.values():
            entry.delete(0, tk.END)
        result_label.config(text="Prediction result will appear here.", fg="#14324a")

    def predict_parkinsons():
        input_data = []

        for feature_name in feature_names:
            value = entries[feature_name].get().strip()
            if not value:
                messagebox.showwarning("Missing value", f"Please enter a value for {feature_name}.")
                entries[feature_name].focus_set()
                return

            try:
                input_data.append(float(value))
            except ValueError:
                messagebox.showerror("Invalid value", f"{feature_name} must be a number.")
                entries[feature_name].focus_set()
                return

        input_frame = pd.DataFrame([input_data], columns=feature_names)
        prediction = model.predict(input_frame)[0]
        probability = model.predict_proba(input_frame)[0]
        confidence = np.max(probability) * 100

        if prediction == 0:
            message = f"Prediction: No Parkinson's detected | Confidence: {confidence:.2f}%"
            result_label.config(text=message, fg="#1f7a5c")
        else:
            message = f"Prediction: Parkinson's detected | Confidence: {confidence:.2f}%"
            result_label.config(text=message, fg="#b42318")

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill="x", pady=(12, 0))

    ttk.Button(button_frame, text="Predict Parkinson's", command=predict_parkinsons, style="Predict.TButton").pack(
        side="left", padx=(0, 8)
    )
    ttk.Button(button_frame, text="Reset Values", command=reset_values).pack(side="left")

    target_message = (
        "Target reached: test accuracy is at least 93%."
        if metrics["test_accuracy"] >= TARGET_ACCURACY
        else "Current test accuracy is below 93%; more data or further tuning may be needed."
    )
    ttk.Label(main_frame, text=target_message, style="Subheader.TLabel").pack(anchor="w", pady=(10, 0))

    root.mainloop()


if __name__ == "__main__":
    trained_model, model_metrics = load_and_train_model()
    create_app(trained_model, model_metrics)
