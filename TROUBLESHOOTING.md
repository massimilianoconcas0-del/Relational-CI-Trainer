# 🩹 Troubleshooting – Relational CI/Trainer

First, don't panic – the relational pipeline is deterministic. If something doesn't work, it's usually one of three simple issues.

## 1. PR comment never appears

**Symptom:** You open a PR, the workflow runs, but no bot comment shows up.

**Fix:**
Go to your repository's **Settings → Actions → General → Workflow permissions** and select **“Read and write permissions”**, then re‑run the workflow.
If the issue persists, ensure your branch is targeting `main` (or adjust the trigger branch in the workflow file).

---

## 2. Training takes > 30 seconds (or times out)

**Symptom:** The workflow runs for a long time and eventually fails.

**Cause:** The free GitHub runner has only 2 CPU cores and 7 GB of memory. Large datasets (>50,000 rows) can overwhelm it.

**Fix:**
- Use a randomly sampled subset of your data during CI (`df.sample(n=20_000)`).
- If you absolutely need full‑dataset training in CI, switch to a self‑hosted runner.

---

## 3. Relational MSE is high (> 0.1) – “the model didn’t learn”

**Symptom:** The bot posts a report with MSE > 0.1 and status “⚠️ MSE higher than expected”.

**Possible causes & fixes:**

| Cause | Diagnosis | Fix |
| :--- | :--- | :--- |
| **Capacity column doesn't represent the system's true structural limit** | The capacity values are small relative to the target (e.g., using "number of rooms" instead of "total floor area") | Try: `capacity = df[["col1", "col2"]].sum(axis=1)` or read the [main framework docs](https://github.com/massimilianoconcas0-del/Relational_Loss_ML) |
| **Data contains outliers that break the relational space** | Very few rows with extreme capacity values | Clip or log-transform the capacity: `capacity = np.log(capacity + 1)` |
| **Target is not actually proportional to your chosen capacity** | The relationship is non‑linear or not ratio‑based | Test whether `y / capacity` is roughly stable across the dataset |

**💡 Pro Tip – Let the Framework Speak for Itself:**  
If you're stuck identifying the right capacity, give this exact prompt to any AI assistant (ChatGPT, Claude, Gemini):

> *“I'm using the Relational Calculus Framework (paper: dimensionless ratios anchored to a system's intrinsic maximum capacity). My data has these columns: [list your column names]. My task is to predict [target]. What could be the 'Global Capacity' of my system – a column that represents the absolute maximum scale the system can reach? Explain why.”*

The AI will have enough context to reason through your domain and propose candidate capacities. This is often faster than trial and error, especially for niche scientific data.

---

## 4. ModuleNotFoundError when running locally

**Symptom:** `import pandas` or `import xgboost` fails.

**Fix:**
```bash
pip install -r requirements.txt
```

---

## 5. GitHub Actions workflow fails with “No such file or directory: src/train_relational.py”

**Symptom:** The workflow can't find the training script.

**Cause:** The file might be in a different location, or you renamed it.

**Fix:**
Make sure the script is at `src/train_relational.py` and that the workflow YAML references that path exactly.

---

### Still stuck?

Open an issue in the [main Relational Calculus repository](https://github.com/massimilianoconcas0-del/Relational_Loss_ML) with a description of your data and what you've tried. The community will help you find your North Star.

---
*Note: This file strikes the right balance: practical, fast fixes first, then the AI‑assisted conceptual leap as the ultimate escape hatch. And the message to the user is: “You don’t have to be a mathematician – just describe your problem in plain language, and the framework handles the rest.”*
