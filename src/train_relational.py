"""
╔══════════════════════════════════════════════════════════════╗
║  CUSTOMIZATION GUIDE (3 lines to change for your data)      ║
║                                                             ║
║  1. Replace generate_absolute_data() with your CSV loader   ║
║  2. Set `capacity` to your domain's "North Star" column     ║
║  3. Update the report.md message with your model name       ║
║                                                             ║
║  See README.md for a full walkthrough with examples.        ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time


def generate_absolute_data(n_samples: int = 5000):
    """
    Demo only: creates synthetic data with high absolute values
    to simulate a real-world scale problem.
    """
    from sklearn.datasets import make_regression

    X_base, y_base = make_regression(
        n_samples=n_samples, n_features=5, noise=0.1, random_state=42
    )
    X_abs = X_base * 150_000
    y_abs = y_base * 80_000 + np.sin(X_base[:, 0]) * 30_000
    return X_abs, y_abs


# ═══════════════════════════════════════════════════════════════
# ── CUSTOMIZE: Replace this block with your data loader ──
X_abs, y_abs = generate_absolute_data()
# ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# ── CUSTOMIZE: Define the North Star capacity for YOUR domain ──
capacity = X_abs.max(axis=1)  # NumPy array, no `.values` needed
# ──────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════

# Relational transformation (no data scaling needed)
# Note: capacity is a 1D numpy array, so we reshape to 2D for division
X_relational = X_abs / capacity.reshape(-1, 1)
y_relational = y_abs / capacity

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_relational, y_relational, test_size=0.2, random_state=42
)

# Train a tiny XGBoost model
model = xgb.XGBRegressor(
    n_estimators=20, max_depth=3, learning_rate=0.3, verbosity=0
)

start = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Write the CI report
with open("report.md", "w") as f:
    f.write(f"## 🤖 Relational‑CI‑Trainer Results\n")
    f.write(f"- **Relational MSE (test):** {mse:.6f}\n")
    f.write(f"- **Training time:** {training_time:.2f} s\n")
    f.write(f"- **Data points:** {len(X_abs)}\n")
    if mse < 0.01:
        f.write(f"- **Status:** ✅ Model successfully learned the relational pattern\n")
    else:
        f.write(f"- **Status:** ⚠️ MSE higher than expected – check capacity column\n")
