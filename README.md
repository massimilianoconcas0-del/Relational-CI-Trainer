# ⚡ Relational CI/Trainer – Instant ML Quality Gates in GitHub Actions

**Train an XGBoost model on a free GitHub runner in under a second – no GPU, no data normalization, no scale fragility.**

[![CI/Trainer Demo](https://github.com/massimilianoconcas0-del/Relational-CI-Trainer/actions/workflows/relational_model.yml/badge.svg)](https://github.com/massimilianoconcas0-del/Relational-CI-Trainer/actions/workflows/relational_model.yml)
![GitHub Actions](https://img.shields.io/badge/Runs_on-GitHub_Actions-2088FF?logo=githubactions)
![XGBoost](https://img.shields.io/badge/Engine-XGBoost-blueviolet)
![Scale Invariant](https://img.shields.io/badge/Property-Scale%20Invariant-brightgreen)
![Optimizer Free](https://img.shields.io/badge/Optimizer-Not_Needed-blue)

---

## 🧠 The Problem
Modern CI pipelines run linters, unit tests, and type checks – but they **never verify if your ML model actually learns**. Why? Because traditional training requires GPUs, data normalization, and complex hyperparameter tuning. That's too slow and too fragile for a CI runner.

**Relational Calculus changes the game.** By training on dimensionless ratios instead of absolute values, we turn model training into a sub‑second, deterministic operation that fits in a GitHub Actions job.

---

## 🔥 What This Template Does
1. **Loads a tiny synthetic dataset** (or your own CSV – see below).  
2. **Transforms absolute numbers into dimensionless relational features** using a single “capacity” column.  
3. **Trains a lightweight XGBoost model in ~0.1 seconds on the free GitHub runner.**  
4. **Posts the Relational MSE (mean squared error) as a comment on your Pull Request.**  

> No GPU, no `MinMaxScaler`, no `Adam` optimizer – just pure geometry.

---

## 🚀 3‑Click Demo (See It Work in 60 Seconds)
1. **Click the green “Use this template” button** at the top of this repo and create your own copy.  
2. **Create a new branch** (`my-test`) and make any dummy edit (e.g., add a comment in `src/train_relational.py`).  
3. **Open a Pull Request** from that branch to `main`.  

The GitHub Action will fire automatically. In ~30 seconds, the **Relational‑CI‑Trainer bot** will comment on your PR with something like:
```text
🤖 Relational‑CI‑Trainer Results
Relational MSE (test): 0.000342
Training time: 0.08 s
Data points: 5000
Status: ✅ Model successfully learned the relational pattern
```

That’s it. You just turned a PR into a model quality gate.

---

## 🔧 Adapt This Template to Your Own Data (Only 3 Lines to Change)
The demo uses synthetic data so you can see the pipeline instantly. To use **your own CSV**, you only need to touch **three lines** in `src/train_relational.py`.

### Step 1: Replace the data loader
Delete the synthetic data generator and load your CSV.
```python
# ❌ DELETE THIS:
# X_abs, y_abs = generate_absolute_data()

# ✅ REPLACE WITH THIS:
import pandas as pd
df = pd.read_csv("your_data.csv")

# Your capacity column (the "North Star" of your system)
capacity = df["your_capacity_column"]   # e.g., "max_budget", "total_power", "housekeeping_sum"

# Features: all columns except target and capacity
feature_cols = [c for c in df.columns if c not in ["target", "your_capacity_column"]]
X_abs = df[feature_cols]
y_abs = df["target"]
```

### Step 2: Define the capacity
The capacity is the intrinsic upper bound of your system – the number that makes all other numbers relative.

Examples:

| Domain | Capacity column (examples) |
| :--- | :--- |
| Finance | Maximum price in a lookback window |
| Physics sim | Maximum theoretical thrust / adiabatic flame temp |
| Biology (scRNA) | Sum of housekeeping gene expression per cell |
| Sensor data | Full‑scale reading of the instrument |
| Retail | Store’s total floor area or max historical footfall |

Change this line:
```python
# ❌ OLD (demo):
# capacity = X_abs.max(axis=1)

# ✅ NEW (your domain):
capacity = df["your_capacity_column"]   # or compute it: df[cols].sum(axis=1)
```

### Step 3: Commit, push, and open a PR
Push the changes to a branch, open a PR, and the workflow will:
* Install dependencies in ~5 seconds.
* Train the relational model in ~0.1 seconds.
* Post the Relational MSE as a PR comment.

That’s it. No other changes needed. The pipeline is now guarding your model quality every time you open a PR.

### 🏠 Full Worked Example: Ames Housing Prices
Here’s a complete adaptation for the classic Ames Housing dataset.
We use `Lot_Area` (the structural capacity of the property) as the capacity.
```python
# src/train_relational.py (adapted for Ames Housing)
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

df = pd.read_csv("ames_housing.csv")
capacity = df["Lot_Area"]   # The property's fundamental size limit
feature_cols = [c for c in df.columns if c not in ["Sale_Price", "Lot_Area"]]
X_abs = df[feature_cols]
y_abs = df["Sale_Price"]

# Relational transformation
X_rel = X_abs.div(capacity, axis=0)
y_rel = y_abs / capacity

X_train, X_test, y_train, y_test = train_test_split(X_rel, y_rel, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=20, max_depth=3, learning_rate=0.3, verbosity=0)
start = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Write report for the PR comment
with open("report.md", "w") as f:
    f.write(f"## 🤖 Relational‑CI‑Trainer Results\n")
    f.write(f"- **Relational MSE (test):** {mse:.6f}\n")
    f.write(f"- **Training time:** {training_time:.2f} s\n")
    f.write(f"- **Data points:** {len(df)}\n")
    if mse < 0.01:
        f.write(f"- **Status:** ✅ Model successfully learned the relational pattern\n")
    else:
        f.write(f"- **Status:** ⚠️ MSE higher than expected – check capacity column\n")
```

## 🧪 Troubleshooting

| Symptom | Likely Cause | Fix |
| :--- | :--- | :--- |
| PR comment never appears | Workflow has no write permissions | Go to Settings → Actions → General → Workflow permissions → Read and write permissions |
| Training took > 30 seconds | Dataset too large for free runner (2‑core CPU, 7 GB RAM) | Reduce to < 50,000 rows or use a representative sample |
| Relational MSE is high (> 0.1) | Capacity column doesn’t capture the true structural boundary | Try `capacity = df[["col_a", "col_b"]].sum(axis=1)` or check the main Relational Calculus docs |
| ModuleNotFoundError: pandas | Dependencies not installed locally | Run `pip install pandas numpy xgboost scikit-learn` before pushing |

## 🧬 The Science Behind It
This template is powered by the **Relational Calculus Framework** – a dimensionless paradigm for machine learning.
Instead of forcing models to memorize absolute units (dollars, meters, counts), it trains them on pure proportions.
The result: models that are immune to data drift, converge thousands of times faster, and generalize across scales without retraining.
See the main repository for deep‑learning examples, physics simulations, and scientific papers:
👉 github.com/massimilianoconcas0-del/Relational_Loss_ML

## 🤝 Contribute
This is a living template. If you adapt it to a new domain (finance, healthcare, IoT), open an issue or PR – we’ll happily link to your example.
* **Star this repo** if it saved you a GPU cycle.
* **Share your adaptation** – the Adapt to Your Data section is meant to grow.
* **Join the main project** to push the limits of dimensionless ML.



## 🔗 Powered By
This template is a technical demonstration of the **Relational Calculus Framework**. 
Discover how to apply this math to solve the "Batch Effect" in Deep Learning, optimize Fluid Dynamics, or achieve Zero-Shot Transfer across entire domains:

👉 **[Explore the main repository: Relational_Loss_ML](https://github.com/massimilianoconcas0-del/Relational_Loss_ML)**

Built with ❤️ by Ciber Fabbrica – because CI should care about model quality, not just code style.
