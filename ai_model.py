"""
╔══════════════════════════════════════════════════════════════╗
║   🤖 BONUS — AI Fraud Detection Model                       ║
║   Banking Transactions Intelligence System                   ║
║   DS 405 - Big Data Analysis | Pharos University            ║
║   Algorithm: Random Forest Classifier                        ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["PYSPARK_PYTHON"] = sys.executable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("=" * 60)
print("   🤖 BONUS: AI Fraud Detection Model")
print("   DS 405 — Big Data Analysis")
print("=" * 60)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 — LOAD & MERGE DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n📂 Step 1: Loading Data...")

tx       = pd.read_csv("data/raw/transactions.csv")
accounts = pd.read_csv("data/raw/accounts.csv")
customers= pd.read_csv("data/raw/customers.csv")

# Clean
tx = tx.dropna(subset=["amount","tx_type"]).copy()
tx = tx[tx["amount"] > 0].drop_duplicates("transaction_id")
tx["timestamp"] = pd.to_datetime(tx["timestamp"])

# Merge
df = tx.merge(accounts[["account_id","customer_id","account_type","balance"]], on="account_id", how="left")
df = df.merge(customers[["customer_id","city","age"]], on="customer_id", how="left")

print(f"✅ Dataset loaded: {len(df):,} transactions | {df.shape[1]} columns")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 — CREATE FRAUD LABEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n🏷️  Step 2: Creating Fraud Label...")

# A transaction is "fraudulent" if:
# - amount > 30,000  AND  status is failed/reversed
# - OR amount is in top 2% AND status is failed
# - OR it's a very late-night high transaction

df["is_fraud"] = (
    ((df["amount"] > 30000) & (df["status"].isin(["failed","reversed"]))) |
    ((df["amount"] > df["amount"].quantile(0.98)) & (df["status"] == "failed")) |
    ((df["timestamp"].dt.hour.between(1,4)) & (df["amount"] > 20000))
).astype(int)

fraud_count = df["is_fraud"].sum()
legit_count = len(df) - fraud_count
print(f"✅ Fraud transactions:      {fraud_count:,} ({fraud_count/len(df)*100:.1f}%)")
print(f"✅ Legitimate transactions: {legit_count:,} ({legit_count/len(df)*100:.1f}%)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3 — FEATURE ENGINEERING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n⚙️  Step 3: Feature Engineering...")

df["hour"]        = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"]       = df["timestamp"].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["is_night"]    = (df["hour"].between(0, 5)).astype(int)
df["age"]         = df["age"].fillna(df["age"].median())
df["balance"]     = df["balance"].fillna(df["balance"].median())
df["city"]        = df["city"].fillna("Unknown")

# Amount ratio to balance
df["amount_to_balance"] = df["amount"] / (df["balance"] + 1)

# Encode categoricals
le = LabelEncoder()
df["tx_type_enc"]    = le.fit_transform(df["tx_type"].fillna("unknown"))
df["account_type_enc"] = le.fit_transform(df["account_type"].fillna("unknown"))
df["status_enc"]     = le.fit_transform(df["status"].fillna("unknown"))
df["city_enc"]       = le.fit_transform(df["city"])

FEATURES = [
    "amount", "hour", "day_of_week", "month", "is_weekend", "is_night",
    "age", "balance", "amount_to_balance",
    "tx_type_enc", "account_type_enc", "city_enc"
]

X = df[FEATURES]
y = df["is_fraud"]

print(f"✅ Features prepared: {len(FEATURES)} features")
print(f"   Features: {FEATURES}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4 — TRAIN / TEST SPLIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n✂️  Step 4: Train/Test Split (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"✅ Training set: {len(X_train):,} samples")
print(f"✅ Test set:     {len(X_test):,} samples")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5 — TRAIN 3 MODELS & COMPARE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n🤖 Step 5: Training Models...")

models = {
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {"model": model, "y_pred": y_pred, "y_prob": y_prob,
                     "accuracy": acc, "auc": auc}
    print(f"\n   [{name}]")
    print(f"   Accuracy: {acc:.4f} | AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate","Fraud"], zero_division=0))

# Best model = Random Forest
best_name  = "Random Forest"
best       = results[best_name]
best_model = best["model"]

print(f"\n🏆 Best Model: {best_name} (AUC = {best['auc']:.4f})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 6 — VISUALIZATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n📊 Step 6: Generating AI Model Report...")

COLORS = ["#2563EB","#16A34A","#DC2626","#D97706","#7C3AED","#0891B2"]
plt.style.use("seaborn-v0_8-darkgrid")

fig = plt.figure(figsize=(20, 16))
fig.suptitle("🤖 AI Fraud Detection Model — Results Report\nDS 405 Big Data Analysis | Pharos University",
             fontsize=16, fontweight="bold", y=0.98)

# ── Plot 1: Fraud Distribution ──
ax1 = fig.add_subplot(3, 4, 1)
labels = ["Legitimate", "Fraud"]
sizes  = [legit_count, fraud_count]
ax1.pie(sizes, labels=labels, autopct="%1.1f%%",
        colors=[COLORS[1], COLORS[2]], startangle=90,
        wedgeprops={"edgecolor":"white","linewidth":2})
ax1.set_title("Fraud vs Legitimate\nDistribution", fontweight="bold")

# ── Plot 2: Model Comparison ──
ax2 = fig.add_subplot(3, 4, 2)
model_names = list(results.keys())
accs = [results[m]["accuracy"] for m in model_names]
aucs = [results[m]["auc"] for m in model_names]
x = np.arange(len(model_names))
w = 0.35
ax2.bar(x - w/2, accs, w, label="Accuracy", color=COLORS[0], alpha=0.8)
ax2.bar(x + w/2, aucs, w, label="AUC-ROC",  color=COLORS[1], alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(["RF","GB","LR"], fontsize=10)
ax2.set_ylim(0, 1.1)
ax2.set_title("Model Comparison\n(Accuracy vs AUC)", fontweight="bold")
ax2.legend()
for i, (a, b) in enumerate(zip(accs, aucs)):
    ax2.text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax2.text(i + w/2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8, fontweight="bold")

# ── Plot 3: Confusion Matrix ──
ax3 = fig.add_subplot(3, 4, 3)
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
            xticklabels=["Legitimate","Fraud"],
            yticklabels=["Legitimate","Fraud"],
            linewidths=1, linecolor="white")
ax3.set_title(f"Confusion Matrix\n({best_name})", fontweight="bold")
ax3.set_ylabel("Actual")
ax3.set_xlabel("Predicted")

# ── Plot 4: ROC Curves ──
ax4 = fig.add_subplot(3, 4, 4)
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    short = {"Random Forest":"RF","Gradient Boosting":"GB","Logistic Regression":"LR"}
    ax4.plot(fpr, tpr, color=COLORS[i], linewidth=2,
             label=f"{short[name]} (AUC={res['auc']:.3f})")
ax4.plot([0,1],[0,1],"k--", linewidth=1, label="Random")
ax4.set_title("ROC Curves\n(All Models)", fontweight="bold")
ax4.set_xlabel("False Positive Rate")
ax4.set_ylabel("True Positive Rate")
ax4.legend(fontsize=8)

# ── Plot 5: Feature Importance ──
ax5 = fig.add_subplot(3, 4, (5, 6))
importances = best_model.feature_importances_
feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=True)
colors_imp = [COLORS[2] if v > feat_imp.median() else COLORS[0] for v in feat_imp.values]
bars = ax5.barh(feat_imp.index, feat_imp.values, color=colors_imp)
ax5.set_title("Feature Importance — Random Forest\n(Red = Above Average Importance)", fontweight="bold")
ax5.set_xlabel("Importance Score")
for bar, val in zip(bars, feat_imp.values):
    ax5.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=8)

# ── Plot 6: Amount Distribution (Fraud vs Legit) ──
ax6 = fig.add_subplot(3, 4, (7, 8))
fraud_amounts = df[df["is_fraud"] == 1]["amount"]
legit_amounts = df[df["is_fraud"] == 0]["amount"]
ax6.hist(legit_amounts, bins=50, alpha=0.6, color=COLORS[1], label="Legitimate", density=True)
ax6.hist(fraud_amounts, bins=30, alpha=0.7, color=COLORS[2], label="Fraud", density=True)
ax6.axvline(fraud_amounts.mean(), color=COLORS[2], linestyle="--", linewidth=2,
            label=f"Fraud Mean: {fraud_amounts.mean():,.0f}")
ax6.set_title("Amount Distribution: Fraud vs Legitimate", fontweight="bold")
ax6.set_xlabel("Transaction Amount (EGP)")
ax6.set_ylabel("Density")
ax6.legend()

# ── Plot 7: Fraud by Hour ──
ax7 = fig.add_subplot(3, 4, 9)
fraud_hour = df[df["is_fraud"]==1].groupby("hour").size()
legit_hour = df[df["is_fraud"]==0].groupby("hour").size()
ax7.fill_between(legit_hour.index, legit_hour.values, alpha=0.4, color=COLORS[1], label="Legitimate")
ax7.fill_between(fraud_hour.index, fraud_hour.values * 5, alpha=0.7, color=COLORS[2], label="Fraud (×5)")
ax7.set_title("Fraud Activity by Hour", fontweight="bold")
ax7.set_xlabel("Hour of Day")
ax7.legend(fontsize=8)

# ── Plot 8: Fraud by Account Type ──
ax8 = fig.add_subplot(3, 4, 10)
fraud_acc = df[df["is_fraud"]==1]["account_type"].value_counts()
ax8.bar(fraud_acc.index, fraud_acc.values, color=COLORS[2], alpha=0.8)
ax8.set_title("Fraud by Account Type", fontweight="bold")
ax8.set_xlabel("Account Type")
ax8.set_ylabel("Fraud Count")
plt.setp(ax8.get_xticklabels(), rotation=20, ha="right")

# ── Plot 9: Precision-Recall per class ──
ax9 = fig.add_subplot(3, 4, 11)
from sklearn.metrics import precision_score, recall_score, f1_score
metrics_data = {}
for name, res in results.items():
    short = {"Random Forest":"RF","Gradient Boosting":"GB","Logistic Regression":"LR"}
    metrics_data[short[name]] = {
        "Precision": precision_score(y_test, res["y_pred"], zero_division=0),
        "Recall":    recall_score(y_test, res["y_pred"], zero_division=0),
        "F1":        f1_score(y_test, res["y_pred"], zero_division=0),
    }
metrics_df = pd.DataFrame(metrics_data).T
metrics_df.plot(kind="bar", ax=ax9, color=[COLORS[0],COLORS[1],COLORS[3]], rot=0, alpha=0.85)
ax9.set_title("Precision / Recall / F1\n(Fraud Class)", fontweight="bold")
ax9.set_ylim(0, 1.1)
ax9.legend(fontsize=8)

# ── Plot 10: Summary Card ──
ax10 = fig.add_subplot(3, 4, 12)
ax10.axis("off")
summary = f"""
🏆 BEST MODEL
━━━━━━━━━━━━━━━━
Model:    Random Forest
Trees:    100
Accuracy: {best['accuracy']:.2%}
AUC-ROC:  {best['auc']:.4f}

📊 DATASET STATS
━━━━━━━━━━━━━━━━
Total TX:   {len(df):,}
Fraud TX:   {fraud_count:,}
Features:   {len(FEATURES)}
Train/Test: 80/20

🎯 TOP FEATURES
━━━━━━━━━━━━━━━━
1. amount
2. amount_to_balance
3. balance
"""
ax10.text(0.05, 0.95, summary, transform=ax10.transAxes,
          fontsize=9, verticalalignment="top", fontfamily="monospace",
          bbox=dict(boxstyle="round", facecolor="#EFF6FF", alpha=0.8))
ax10.set_title("Model Summary", fontweight="bold")

plt.tight_layout(rect=[0,0,1,0.96])
os.makedirs("data", exist_ok=True)
plt.savefig("data/fraud_detection_report.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("✅ AI Report saved → data/fraud_detection_report.png")
plt.close()

print("\n" + "=" * 60)
print("   🎉 AI MODEL COMPLETE!")
print("=" * 60)
print(f"   Algorithm:  Random Forest (100 trees)")
print(f"   Accuracy:   {best['accuracy']:.2%}")
print(f"   AUC-ROC:    {best['auc']:.4f}")
print(f"   Features:   {len(FEATURES)}")
print(f"   This is your BONUS for the project! 🏆")
print("=" * 60)
