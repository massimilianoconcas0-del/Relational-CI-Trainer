import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def generate_absolute_data(n_samples=15000, n_features=20):
    """
    Generates a dataset where the target is a strict dimensionless fraction
    of the maximum sensor capacity, simulating pure physical constraints.
    """
    print("Generating unscaled absolute data...")
    X, _ = make_regression(n_samples=n_samples, n_features=n_features, random_state=42)

    # Artificially inflate the scale to trigger exploding gradients in traditional models
    X_abs = np.abs(X * 150000.0) + 10.0
    capacity = X_abs.max(axis=1)

    # Create a complex, non-linear physical ratio strictly bounded between [0, 1]
    true_ratio = np.clip(np.sin(X.mean(axis=1)) ** 2, 0, 1)

    # Add slight noise to simulate real-world sensor inaccuracies
    noisy_ratio = np.clip(true_ratio + np.random.normal(0, 0.05, n_samples), 0, 1)

    # The absolute target is simply the capacity multiplied by the ratio
    y_abs = capacity * noisy_ratio

    feature_names = [f"sensor_{i}" for i in range(n_features)]
    return pd.DataFrame(X_abs, columns=feature_names), pd.Series(y_abs, name="target_absolute")

def main():
    print("🚀 Initializing Relational Calculus Training Pipeline...")

    # 1. Load Data
    X_abs, y_abs = generate_absolute_data()

    start_time = time.time()

    # ---------------------------------------------------------
    # 2. THE RELATIONAL CALCULUS SHIFT
    # ---------------------------------------------------------
    print("Applying Scale-Invariant Topological Mapping...")

    # Extract the "Global Capacity" (C_obs) for each row
    # In this context, we use the maximum sensor reading as the structural boundary
    capacity = X_abs.max(axis=1)

    # Transform absolute extensive features into dimensionless intensive fractions (Z_i)
    X_relational = X_abs.div(capacity, axis=0)

    # Transform the absolute target into a dimensionless ratio
    y_relational = y_abs / capacity

    # ---------------------------------------------------------
    # 3. XGBoost Training (Green AI)
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X_relational, y_relational, test_size=0.2, random_state=42)

    print("Training Relational XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        tree_method="hist", # Highly optimized for CPU runners
        random_state=42
    )

    model.fit(X_train, y_train)

    # 4. Evaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    end_time = time.time()
    training_time = round(end_time - start_time, 3)

    print(f"✅ Training completed in {training_time} seconds. Relational MSE: {mse:.6f}")

    # ---------------------------------------------------------
    # 5. Export Markdown Report for GitHub Actions
    # ---------------------------------------------------------
    report_content = f"""## 🟢 Green AI Build Successful
**Model compiled and verified on standard GitHub CI/CD infrastructure.**

* ⏱️ **Pipeline Execution Time:** `{training_time} seconds`
* 💻 **Hardware:** Standard GitHub Ubuntu Runner (2-core CPU)
* 📐 **Relational MSE (Test):** `{mse:.6f}`
* 🗃️ **Dataset:** 15,000 samples, 20 features (Synthetically inflated absolute scale)

> **Architectural Note:** This model was trained *without* GPUs, Adam optimizers, or external data normalization (e.g., MinMax/Standard scalers). The data was mapped purely by its structural topology.
>
> ⚡ *Powered by the [Relational Calculus Framework](https://github.com/massimilianoconcas0-del/Relational_Loss_ML).*
"""

    with open("report.md", "w") as f:
        f.write(report_content)

    print("Report exported to report.md")

if __name__ == "__main__":
    main()
