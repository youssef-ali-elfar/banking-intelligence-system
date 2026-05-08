"""
Phase 8 (Bonus): Visualizations
Banking Transactions Intelligence System
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Style ──
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED", "#0891B2", "#DB2777", "#65A30D"]
sns.set_palette(COLORS)

# ── Load clean data ──
tx = pd.read_csv("data/raw/transactions.csv")
customers = pd.read_csv("data/raw/customers.csv")
accounts = pd.read_csv("data/raw/accounts.csv")

tx = tx.dropna(subset=["amount", "tx_type"]).copy()
tx = tx[tx["amount"] > 0].drop_duplicates("transaction_id")
tx["timestamp"] = pd.to_datetime(tx["timestamp"])
tx["month"] = tx["timestamp"].dt.month
tx["hour"] = tx["timestamp"].dt.hour
tx["month_name"] = tx["timestamp"].dt.strftime("%b")

# Merge
merged = tx.merge(accounts[["account_id", "customer_id", "account_type"]], on="account_id", how="left")
merged = merged.merge(customers[["customer_id", "city"]], on="customer_id", how="left")
merged["city"] = merged["city"].fillna("Unknown")

fig = plt.figure(figsize=(20, 22))
fig.suptitle("Banking Transactions Intelligence System\nDS 405 — Big Data Analysis | Pharos University",
             fontsize=18, fontweight="bold", y=0.98)

# ── Plot 1: Transactions by Category ──
ax1 = fig.add_subplot(4, 3, 1)
cat_counts = merged.groupby("category")["amount"].sum().sort_values(ascending=True)
bars = ax1.barh(cat_counts.index, cat_counts.values / 1e6, color=COLORS)
ax1.set_title("Total Spend by Category (EGP M)", fontweight="bold")
ax1.set_xlabel("Amount (Millions)")
for bar, val in zip(bars, cat_counts.values):
    ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f"{val/1e6:.1f}M", va="center", fontsize=8)

# ── Plot 2: Transactions by Type ──
ax2 = fig.add_subplot(4, 3, 2)
type_counts = merged["tx_type"].value_counts()
wedges, texts, autotexts = ax2.pie(type_counts.values, labels=type_counts.index,
                                    autopct="%1.1f%%", colors=COLORS[:len(type_counts)],
                                    startangle=90)
ax2.set_title("Transaction Types Distribution", fontweight="bold")

# ── Plot 3: Monthly Transaction Volume ──
ax3 = fig.add_subplot(4, 3, 3)
monthly = merged.groupby("month").agg(count=("transaction_id","count"), total=("amount","sum")).reset_index()
ax3.bar(monthly["month"], monthly["count"], color=COLORS[0], alpha=0.7, label="Count")
ax3_twin = ax3.twinx()
ax3_twin.plot(monthly["month"], monthly["total"]/1e6, color=COLORS[2], marker="o", linewidth=2, label="Amount (M)")
ax3.set_title("Monthly Volume & Amount", fontweight="bold")
ax3.set_xlabel("Month")
ax3.set_ylabel("Transaction Count", color=COLORS[0])
ax3_twin.set_ylabel("Amount (EGP M)", color=COLORS[2])
ax3.set_xticks(range(1, 13))

# ── Plot 4: Amount Distribution (Histogram) ──
ax4 = fig.add_subplot(4, 3, 4)
ax4.hist(merged["amount"], bins=50, color=COLORS[1], edgecolor="white", alpha=0.8)
ax4.axvline(merged["amount"].mean(), color=COLORS[2], linestyle="--", linewidth=2, label=f'Mean: {merged["amount"].mean():,.0f}')
ax4.set_title("Transaction Amount Distribution", fontweight="bold")
ax4.set_xlabel("Amount (EGP)")
ax4.set_ylabel("Frequency")
ax4.legend()

# ── Plot 5: Spend per City ──
ax5 = fig.add_subplot(4, 3, 5)
city_spend = merged.groupby("city")["amount"].sum().sort_values(ascending=False)
ax5.bar(city_spend.index, city_spend.values / 1e6, color=COLORS)
ax5.set_title("Total Spend per City (EGP M)", fontweight="bold")
ax5.set_xlabel("City")
ax5.set_ylabel("Amount (Millions)")
plt.setp(ax5.get_xticklabels(), rotation=30, ha="right")

# ── Plot 6: Transaction Status ──
ax6 = fig.add_subplot(4, 3, 6)
status_counts = merged["status"].value_counts()
colors_status = [COLORS[1], COLORS[3], COLORS[2], COLORS[0]]
ax6.bar(status_counts.index, status_counts.values, color=colors_status)
ax6.set_title("Transaction Status Breakdown", fontweight="bold")
ax6.set_xlabel("Status")
ax6.set_ylabel("Count")
for i, v in enumerate(status_counts.values):
    ax6.text(i, v + 5, str(v), ha="center", fontweight="bold")

# ── Plot 7: Hourly Activity Heatmap ──
ax7 = fig.add_subplot(4, 3, 7)
merged["day_of_week"] = merged["timestamp"].dt.dayofweek
heatmap_data = merged.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
heatmap_data.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
sns.heatmap(heatmap_data, ax=ax7, cmap="Blues", linewidths=0.1, cbar_kws={"shrink": 0.8})
ax7.set_title("Transaction Activity Heatmap\n(Day × Hour)", fontweight="bold")
ax7.set_xlabel("Hour of Day")
ax7.set_ylabel("")

# ── Plot 8: Account Type Distribution ──
ax8 = fig.add_subplot(4, 3, 8)
acc_type = merged["account_type"].value_counts()
ax8.bar(acc_type.index, acc_type.values, color=COLORS[:4])
ax8.set_title("Transactions by Account Type", fontweight="bold")
ax8.set_xlabel("Account Type")
ax8.set_ylabel("Count")
plt.setp(ax8.get_xticklabels(), rotation=20, ha="right")

# ── Plot 9: Amount Category ──
ax9 = fig.add_subplot(4, 3, 9)
merged["amount_cat"] = pd.cut(merged["amount"],
    bins=[0, 500, 5000, 20000, float("inf")],
    labels=["Small\n(<500)", "Medium\n(500-5K)", "Large\n(5K-20K)", "Very Large\n(>20K)"]
)
cat_dist = merged["amount_cat"].value_counts().sort_index()
wedges, texts, autotexts = ax9.pie(cat_dist.values, labels=cat_dist.index,
                                    autopct="%1.1f%%", colors=COLORS[:4], startangle=45)
ax9.set_title("Amount Category Distribution", fontweight="bold")

# ── Plot 10: Suspicious Transactions ──
ax10 = fig.add_subplot(4, 3, 10)
suspicious = merged[(merged["amount"] > 30000) | (merged["status"].isin(["failed", "reversed"]))]
susp_city = suspicious.groupby("city").size().sort_values(ascending=False)
ax10.bar(susp_city.index, susp_city.values, color=COLORS[2], alpha=0.8)
ax10.set_title("⚠️ Suspicious Transactions per City", fontweight="bold")
ax10.set_xlabel("City")
ax10.set_ylabel("Count")
plt.setp(ax10.get_xticklabels(), rotation=30, ha="right")

# ── Plot 11: Top 10 Customers by Spend ──
ax11 = fig.add_subplot(4, 3, 11)
top_customers = merged.groupby("customer_id")["amount"].sum().nlargest(10).reset_index()
top_customers = top_customers.merge(customers[["customer_id","name"]], on="customer_id")
ax11.barh(top_customers["name"], top_customers["amount"]/1e3, color=COLORS[4])
ax11.set_title("Top 10 Customers by Total Spend", fontweight="bold")
ax11.set_xlabel("Amount (EGP 000s)")
ax11.invert_yaxis()

# ── Plot 12: Data Quality Summary ──
ax12 = fig.add_subplot(4, 3, 12)
tx_raw = pd.read_csv("data/raw/transactions.csv")
quality_labels = ["Valid\nRecords", "Null\nAmounts", "Negative\nAmounts", "Null\nTx Type", "Duplicates"]
quality_vals = [
    len(tx_raw) - tx_raw["amount"].isna().sum() - (tx_raw["amount"] < 0).sum(),
    tx_raw["amount"].isna().sum(),
    (tx_raw["amount"] < 0).sum(),
    tx_raw["tx_type"].isna().sum(),
    tx_raw.duplicated("transaction_id").sum()
]
quality_colors = [COLORS[1], COLORS[2], COLORS[2], COLORS[3], COLORS[3]]
bars = ax12.bar(quality_labels, quality_vals, color=quality_colors)
ax12.set_title("Data Quality Report\n(Before Cleaning)", fontweight="bold")
ax12.set_ylabel("Record Count")
for bar, val in zip(bars, quality_vals):
    ax12.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
              str(val), ha="center", fontweight="bold", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("data/banking_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("✅ Dashboard saved → data/banking_dashboard.png")
plt.close()
