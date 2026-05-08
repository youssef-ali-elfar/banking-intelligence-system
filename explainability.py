"""
╔══════════════════════════════════════════════════════════════╗
║   🔍 AI Explainability — Why did the model flag this?       ║
║   Using SHAP (SHapley Additive exPlanations)                ║
║   DS 405 - Big Data Analysis | Pharos University            ║
╚══════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import io
# Force UTF-8 output on Windows so emoji characters print correctly
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# pyrefly: ignore [missing-import]
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("=" * 60)
print("   🔍 AI Explainability — SHAP Analysis")
print("   Answering: WHY did the model flag fraud?")
print("=" * 60)

# ── Load & Prepare ──
tx       = pd.read_csv("data/raw/transactions.csv")
accounts = pd.read_csv("data/raw/accounts.csv")
customers= pd.read_csv("data/raw/customers.csv")

tx = tx.dropna(subset=["amount","tx_type"]).copy()
tx = tx[tx["amount"] > 0].drop_duplicates("transaction_id")
tx["timestamp"] = pd.to_datetime(tx["timestamp"])

df = tx.merge(accounts[["account_id","customer_id","account_type","balance"]], on="account_id", how="left")
df = df.merge(customers[["customer_id","city","age"]], on="customer_id", how="left")

df["is_fraud"] = (
    ((df["amount"] > 30000) & (df["status"].isin(["failed","reversed"]))) |
    ((df["amount"] > df["amount"].quantile(0.98)) & (df["status"] == "failed")) |
    ((df["timestamp"].dt.hour.between(1,4)) & (df["amount"] > 20000))
).astype(int)

df["hour"]        = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"]       = df["timestamp"].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["is_night"]    = (df["hour"].between(0, 5)).astype(int)
df["age"]         = df["age"].fillna(df["age"].median())
df["balance"]     = df["balance"].fillna(df["balance"].median())
df["city"]        = df["city"].fillna("Unknown")
df["amount_to_balance"] = df["amount"] / (df["balance"] + 1)

le = LabelEncoder()
df["tx_type_enc"]      = le.fit_transform(df["tx_type"].fillna("unknown"))
df["account_type_enc"] = le.fit_transform(df["account_type"].fillna("unknown"))
df["city_enc"]         = le.fit_transform(df["city"])

FEATURES = ["amount","hour","day_of_week","month","is_weekend","is_night",
            "age","balance","amount_to_balance","tx_type_enc","account_type_enc","city_enc"]
FEAT_LABELS = ["Amount","Hour","Day of Week","Month","Is Weekend","Is Night",
               "Customer Age","Account Balance","Amount/Balance Ratio",
               "TX Type","Account Type","City"]

X = df[FEATURES]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── Train Model ──
print("\n🤖 Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
print("✅ Model trained")

# ── SHAP Explainer ──
print("\n🔍 Computing SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
# Handle possible output shapes from SHAP
if isinstance(shap_values, list):
    # Binary classification returns list of arrays per class
    shap_fraud = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    # Shape (samples, features, classes) – take fraud class (index 1)
    shap_fraud = shap_values[:, :, 1]
else:
    shap_fraud = shap_values

print("shap_fraud shape:", shap_fraud.shape)  # debug

# Compute mean absolute SHAP values per feature (ensure 1‑D)
mean_abs_shap = np.abs(shap_fraud).mean(axis=0)
if mean_abs_shap.ndim > 1:
    # Collapse any extra dimensions (e.g., (features, 2))
    mean_abs_shap = mean_abs_shap.ravel()


print("✅ SHAP values computed")

# ── Pick examples to explain ──
fraud_indices = X_test[y_test == 1].index[:3].tolist()
legit_indices = X_test[y_test == 0].index[:2].tolist()
explain_indices = fraud_indices + legit_indices

print(f"\n📋 Explaining {len(explain_indices)} transactions:")
print(f"   {len(fraud_indices)} fraud + {len(legit_indices)} legitimate")


# ── Visualization ──
COLORS = ["#ef4444","#3b82f6","#22c55e","#f59e0b","#a855f7"]
plt.style.use("seaborn-v0_8-darkgrid")

fig = plt.figure(figsize=(20, 18))
fig.suptitle("🔍 AI Explainability Report — Why Did the Model Decide?\nDS 405 Big Data Analysis | Pharos University",
             fontsize=15, fontweight="bold", y=0.99)

# ── Plot 1: Global Feature Importance (SHAP mean abs) ──
ax1 = fig.add_subplot(3, 3, (1,2))
mean_abs_shap = np.abs(shap_fraud).mean(axis=0)

# Flatten to 1-D regardless of shap_fraud shape
if mean_abs_shap.ndim == 2:
    mean_abs_shap = mean_abs_shap[:, 1]   # column 1 = fraud class
elif mean_abs_shap.ndim > 2:
    mean_abs_shap = mean_abs_shap.mean(axis=-1)

print(mean_abs_shap.shape)   # should print (12,)
feat_shap = pd.Series(mean_abs_shap, index=FEAT_LABELS).sort_values(ascending=True)
colors_bar = ["#ef4444" if v > feat_shap.median() else "#3b82f6" for v in feat_shap.values]
bars = ax1.barh(feat_shap.index, feat_shap.values, color=colors_bar, edgecolor="none")
ax1.set_title("Global Feature Impact on Fraud Detection\n(Mean |SHAP Value| — Higher = More Important)",
              fontweight="bold")
ax1.set_xlabel("Mean |SHAP Value|")
for bar, val in zip(bars, feat_shap.values):
    ax1.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=8)
red_p  = mpatches.Patch(color="#ef4444", label="Above avg importance")
blue_p = mpatches.Patch(color="#3b82f6", label="Below avg importance")
ax1.legend(handles=[red_p, blue_p], fontsize=8)

# ── Plot 2: SHAP Summary Scatter ──
ax2 = fig.add_subplot(3, 3, 3)
# Scatter of amount SHAP vs amount value
x_vals = X_test["amount"].values
y_vals = shap_fraud[:, 0]  # SHAP for amount
scatter_colors = ["#ef4444" if y_test.iloc[i]==1 else "#3b82f6"
                  for i in range(len(y_test))]
ax2.scatter(x_vals, y_vals, c=scatter_colors, alpha=0.5, s=15)
ax2.axhline(0, color="white", linewidth=1, linestyle="--", alpha=0.5)
ax2.set_title("SHAP Impact of Amount\n(Red=Fraud, Blue=Legit)", fontweight="bold")
ax2.set_xlabel("Transaction Amount (EGP)")
ax2.set_ylabel("SHAP Value (Impact on Fraud Score)")

# ── Plot 3-7: Individual Transaction Explanations ──
test_positions = {idx: pos for pos, idx in enumerate(X_test.index)}

for i, orig_idx in enumerate(explain_indices):
    ax = fig.add_subplot(3, 3, 4 + i)

    # Find position in X_test
    pos = list(X_test.index).index(orig_idx)
    shap_row = shap_fraud[pos]
    feat_vals = X_test.iloc[pos]
    actual    = y_test.iloc[pos]
    pred_prob = model.predict_proba(X_test.iloc[[pos]])[0][1]

    # Top 6 features by abs SHAP
    top_idx = np.argsort(np.abs(shap_row))[-6:]
    labels  = [FEAT_LABELS[j] for j in top_idx]
    values  = [shap_row[j] for j in top_idx]
    fvals   = [feat_vals.iloc[j] for j in top_idx]

    bar_colors = ["#ef4444" if v > 0 else "#22c55e" for v in values]
    ax.barh(labels, values, color=bar_colors, edgecolor="none")
    ax.axvline(0, color="white", linewidth=1)

    verdict = "🚨 FRAUD" if actual == 1 else "✅ LEGIT"
    color   = "#ef4444" if actual == 1 else "#22c55e"
    ax.set_title(f"TX #{orig_idx} — {verdict}\nProb: {pred_prob:.1%}",
                 fontweight="bold", color=color, fontsize=9)
    ax.set_xlabel("SHAP Value", fontsize=8)
    ax.tick_params(labelsize=7)

    # Annotate feature values
    for j, (label, val, fval) in enumerate(zip(labels, values, fvals)):
        ax.text(val + (0.001 if val >= 0 else -0.001),
                j, f"={fval:.1f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=6.5, color="white")

plt.tight_layout(rect=[0,0,1,0.97])
import os; os.makedirs("data", exist_ok=True)
plt.savefig("data/explainability_report.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("\n✅ Explainability report saved → data/explainability_report.png")

# ── Print human-readable explanation ──
print("\n" + "="*60)
print("   📖 HUMAN-READABLE EXPLANATIONS")
print("="*60)

for orig_idx in fraud_indices[:2]:
    pos      = list(X_test.index).index(orig_idx)
    shap_row = shap_fraud[pos]
    feat_vals= X_test.iloc[pos]
    pred_prob= model.predict_proba(X_test.iloc[[pos]])[0][1]

    top_pos = np.argsort(shap_row)[-3:][::-1]
    top_neg = np.argsort(shap_row)[:3]

    print(f"\n🚨 Transaction #{orig_idx} — Fraud Score: {pred_prob:.1%}")
    print(f"   Why it was flagged:")
    for j in top_pos:
        if shap_row[j] > 0:
            print(f"   ↑ {FEAT_LABELS[j]} = {feat_vals.iloc[j]:.2f} "
                  f"(pushed fraud score UP by {shap_row[j]:.4f})")
    print(f"   What reduced suspicion:")
    for j in top_neg:
        if shap_row[j] < 0:
            print(f"   ↓ {FEAT_LABELS[j]} = {feat_vals.iloc[j]:.2f} "
                  f"(pushed fraud score DOWN by {abs(shap_row[j]):.4f})")

print("\n" + "="*60)
print("   ✅ EXPLAINABILITY COMPLETE!")
print("   The SHAP values show EXACTLY why each transaction")
print("   was flagged — not just a black-box prediction! 🎯")
print("="*60)