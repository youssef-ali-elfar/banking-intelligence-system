"""
╔══════════════════════════════════════════════════════════════╗
║   Banking Transactions Intelligence System                   ║
║   DS 405 - Big Data Analysis | Pharos University            ║
║   Phases: Cleaning → Features → Joins → Windows → SQL       ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Windows: point Spark at local winutils.exe & hadoop.dll ──
_hadoop_home = os.path.abspath(os.path.join(os.path.dirname(__file__), "hadoop"))
os.environ["HADOOP_HOME"]      = _hadoop_home
os.environ["hadoop.home.dir"]  = _hadoop_home
os.environ["PYSPARK_PYTHON"]   = sys.executable
# Add hadoop/bin to PATH so JVM can load hadoop.dll
_hadoop_bin = os.path.join(_hadoop_home, "bin")
if _hadoop_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _hadoop_bin + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SPARK SESSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
spark = SparkSession.builder \
    .appName("BankingTransactionsIntelligence") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.ui.showConsoleProgress", "false") \
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
    .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("=" * 60)
print("   Banking Transactions Intelligence System")
print("   DS 405 — Big Data Analysis")
print("=" * 60)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 1 — LOAD DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n📂 PHASE 1: Loading Raw Data")
print("-" * 40)

customers_raw = spark.read.csv("data/raw/customers.csv", header=True, inferSchema=True)
accounts_raw  = spark.read.csv("data/raw/accounts.csv",  header=True, inferSchema=True)
transactions_raw = spark.read.csv("data/raw/transactions.csv", header=True, inferSchema=True)

print(f"✅ Customers:    {customers_raw.count():,} rows | {len(customers_raw.columns)} cols")
print(f"✅ Accounts:     {accounts_raw.count():,} rows | {len(accounts_raw.columns)} cols")
print(f"✅ Transactions: {transactions_raw.count():,} rows | {len(transactions_raw.columns)} cols")

print("\n📋 Transactions Schema:")
transactions_raw.printSchema()

print("📋 Dirty Data Preview (first 5 rows):")
transactions_raw.show(5, truncate=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 2 — DATA CLEANING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n🧹 PHASE 2: Data Cleaning")
print("-" * 40)

# ── 2A. Customers ──
print("→ Cleaning customers...")
before = customers_raw.count()
customers_clean = customers_raw \
    .dropna(subset=["customer_id", "name"]) \
    .fillna({"city": "Unknown", "age": 0}) \
    .filter(F.col("age") >= 0) \
    .dropDuplicates(["customer_id"])
print(f"   Before: {before:,} | After: {customers_clean.count():,}")

# ── 2B. Accounts ──
print("→ Cleaning accounts...")
before = accounts_raw.count()
accounts_clean = accounts_raw \
    .dropna(subset=["account_id", "customer_id"]) \
    .fillna({"balance": 0.0}) \
    .filter(F.col("balance") >= 0) \
    .dropDuplicates(["account_id"])
print(f"   Before: {before:,} | After: {accounts_clean.count():,}")

# ── 2C. Transactions ──
print("→ Cleaning transactions...")
before = transactions_raw.count()

# Step 1: Remove nulls in critical fields
tx_clean = transactions_raw.dropna(subset=["transaction_id", "account_id", "amount", "tx_type"])

# Step 2: Remove negative amounts (invalid)
tx_clean = tx_clean.filter(F.col("amount") > 0)

# Step 3: Remove duplicates
tx_clean = tx_clean.dropDuplicates(["transaction_id"])

# Step 4: Fix timestamp column type
tx_clean = tx_clean.withColumn("timestamp", F.to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss"))

after = tx_clean.count()
print(f"   Before: {before:,} | After: {after:,} | Removed: {before - after:,} dirty records")

print("\n✅ Clean Transactions Sample:")
tx_clean.show(5, truncate=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 3 — FEATURE ENGINEERING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n⚙️  PHASE 3: Feature Engineering")
print("-" * 40)

tx_featured = tx_clean \
    .withColumn("hour",         F.hour("timestamp")) \
    .withColumn("day_of_week",  F.dayofweek("timestamp")) \
    .withColumn("month",        F.month("timestamp")) \
    .withColumn("year",         F.year("timestamp")) \
    \
    .withColumn("is_weekend",   (F.col("day_of_week").isin([1, 7])).cast("int")) \
    \
    .withColumn("time_of_day",
        F.when(F.col("hour").between(6,  11), "morning")
         .when(F.col("hour").between(12, 17), "afternoon")
         .when(F.col("hour").between(18, 21), "evening")
         .otherwise("night")
    ) \
    \
    .withColumn("amount_category",
        F.when(F.col("amount") <  500,   "small")
         .when(F.col("amount") <  5000,  "medium")
         .when(F.col("amount") <  20000, "large")
         .otherwise("very_large")
    ) \
    \
    .withColumn("tx_direction",
        F.when(F.col("tx_type").isin(["credit"]),                   "inflow")
         .when(F.col("tx_type").isin(["debit","withdrawal","payment"]), "outflow")
         .otherwise("neutral")
    ) \
    \
    .withColumn("is_suspicious",
        (
            (F.col("amount") > 30000) |
            (F.col("status") == "failed")
        ).cast("int")
    )

print("✅ New Features Added:")
tx_featured.select("transaction_id", "amount", "amount_category",
                    "time_of_day", "tx_direction", "is_weekend",
                    "is_suspicious").show(8, truncate=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 4 — JOINS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n🔗 PHASE 4: Joining 3 Datasets")
print("-" * 40)

# transactions ↔ accounts ↔ customers
full_df = tx_featured \
    .join(accounts_clean.select("account_id", "customer_id", "account_type", "balance"),
          on="account_id", how="left") \
    .join(customers_clean.select("customer_id", "name", "city", "age"),
          on="customer_id", how="left")

print(f"✅ Full Joined Dataset: {full_df.count():,} rows | {len(full_df.columns)} columns")

print("\n📋 Joined Sample:")
full_df.select("transaction_id", "name", "city", "account_type",
               "amount", "tx_type", "amount_category", "status").show(8, truncate=True)

# Register as SQL temp view
full_df.createOrReplaceTempView("banking_data")
tx_featured.createOrReplaceTempView("transactions")
accounts_clean.createOrReplaceTempView("accounts")
customers_clean.createOrReplaceTempView("customers")
print("✅ Temp Views created for SQL phase")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 5 — AGGREGATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n📊 PHASE 5: Aggregations")
print("-" * 40)

# Total spend per customer
spend_per_customer = full_df \
    .filter(F.col("tx_direction") == "outflow") \
    .groupBy("customer_id", "name", "city") \
    .agg(
        F.count("transaction_id").alias("total_transactions"),
        F.round(F.sum("amount"), 2).alias("total_spend"),
        F.round(F.avg("amount"), 2).alias("avg_tx_amount"),
        F.max("amount").alias("max_single_tx")
    ) \
    .orderBy(F.desc("total_spend"))

print("💰 Top 10 Customers by Total Spend:")
spend_per_customer.show(10, truncate=False)

# Transaction volume by category
print("🏷️  Spending by Category:")
full_df.groupBy("category") \
    .agg(
        F.count("*").alias("num_transactions"),
        F.round(F.sum("amount"), 2).alias("total_amount")
    ).orderBy(F.desc("total_amount")).show(truncate=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 6 — WINDOW FUNCTIONS ⭐
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n🪟 PHASE 6: Window Functions")
print("-" * 40)

# ── 6A. RANK customers by total activity ──
window_rank = Window.partitionBy("city").orderBy(F.desc("total_spend"))

ranked_customers = spend_per_customer \
    .withColumn("rank_in_city", F.rank().over(window_rank)) \
    .withColumn("dense_rank",   F.dense_rank().over(window_rank))

print("🏆 Top Ranked Customers per City:")
ranked_customers.filter(F.col("rank_in_city") <= 3) \
    .select("city", "rank_in_city", "name", "total_spend", "total_transactions") \
    .orderBy("city", "rank_in_city") \
    .show(20, truncate=False)

# ── 6B. LAG — Detect Sudden Spikes ──
window_lag = Window.partitionBy("account_id").orderBy("timestamp")

tx_with_lag = full_df \
    .withColumn("prev_amount", F.lag("amount", 1).over(window_lag)) \
    .withColumn("amount_change", F.col("amount") - F.col("prev_amount")) \
    .withColumn("spike_ratio",
        F.when(F.col("prev_amount") > 0,
               F.round(F.col("amount") / F.col("prev_amount"), 2))
         .otherwise(None)
    ) \
    .withColumn("is_spike", (F.col("spike_ratio") > 5).cast("int"))

print("⚠️  Detected Transaction Spikes (ratio > 5x previous):")
tx_with_lag.filter(F.col("is_spike") == 1) \
    .select("account_id", "name", "timestamp", "prev_amount",
            "amount", "spike_ratio") \
    .orderBy(F.desc("spike_ratio")) \
    .show(10, truncate=False)

# ── 6C. Running Total per Account ──
window_running = Window.partitionBy("account_id") \
    .orderBy("timestamp") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

tx_running = full_df \
    .withColumn("running_total", F.round(F.sum("amount").over(window_running), 2)) \
    .withColumn("tx_count_so_far", F.count("transaction_id").over(window_running))

print("💳 Running Total per Account (sample):")
tx_running.select("account_id", "name", "timestamp", "amount", "running_total", "tx_count_so_far") \
    .orderBy("account_id", "timestamp") \
    .show(12, truncate=True)

# ── 6D. PERCENT_RANK of transaction amounts ──
window_pct = Window.orderBy("amount")
tx_with_rank = full_df \
    .withColumn("amount_percentile", F.round(F.percent_rank().over(window_pct) * 100, 1))

print("📈 Top 5 Highest Transactions with Percentile:")
tx_with_rank.select("transaction_id", "name", "amount", "amount_percentile") \
    .orderBy(F.desc("amount")) \
    .show(5, truncate=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 7 — SQL VS DATAFRAME COMPARISON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n🗄️  PHASE 7: SQL vs DataFrame Comparison")
print("-" * 40)

print("→ Query: Fraud-like patterns (high amount + failed/reversed)")

# ── DataFrame API ──
print("\n[DataFrame API]")
fraud_df = full_df.filter(
    (F.col("amount") > 10000) &
    (F.col("status").isin(["failed", "reversed"]))
).select("transaction_id", "name", "city", "amount", "status", "timestamp") \
 .orderBy(F.desc("amount"))
fraud_df.show(8, truncate=False)

# ── Spark SQL ──
print("[Spark SQL — same result]")
fraud_sql = spark.sql("""
    SELECT
        transaction_id,
        name,
        city,
        amount,
        status,
        timestamp
    FROM banking_data
    WHERE amount > 10000
      AND status IN ('failed', 'reversed')
    ORDER BY amount DESC
    LIMIT 8
""")
fraud_sql.show(8, truncate=False)

# ── SQL: Total spend per city ──
print("→ SQL Query: Total spend per city")
spark.sql("""
    SELECT
        city,
        COUNT(*)                    AS num_transactions,
        ROUND(SUM(amount), 2)       AS total_spend,
        ROUND(AVG(amount), 2)       AS avg_amount
    FROM banking_data
    WHERE tx_direction = 'outflow'
    GROUP BY city
    ORDER BY total_spend DESC
""").show(truncate=False)

# ── SQL: Window Function in SQL ──
print("→ SQL Window: Running total + rank per account")
spark.sql("""
    SELECT
        account_id,
        name,
        timestamp,
        amount,
        ROUND(SUM(amount) OVER (
            PARTITION BY account_id
            ORDER BY timestamp
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ), 2) AS running_total,
        RANK() OVER (
            PARTITION BY city
            ORDER BY amount DESC
        ) AS rank_in_city
    FROM banking_data
    ORDER BY account_id, timestamp
    LIMIT 15
""").show(truncate=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 8 — PARTITIONED SAVE (Parquet)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n💾 PHASE 8: Saving Partitioned Data (Parquet)")
print("-" * 40)

output_path = "data/processed"

# Save full dataset partitioned by account_type
full_df.write \
    .mode("overwrite") \
    .partitionBy("account_type") \
    .parquet(f"{output_path}/banking_full")

# Save fraud-like transactions separately
fraud_df.write \
    .mode("overwrite") \
    .parquet(f"{output_path}/suspicious_transactions")

# Save customer rankings
ranked_customers.write \
    .mode("overwrite") \
    .parquet(f"{output_path}/customer_rankings")

print(f"✅ Full dataset saved   → {output_path}/banking_full/ (partitioned by account_type)")
print(f"✅ Suspicious txns saved → {output_path}/suspicious_transactions/")
print(f"✅ Customer rankings     → {output_path}/customer_rankings/")

# Verify partitions
import os
partitions = [d for d in os.listdir(f"{output_path}/banking_full") if d.startswith("account_type=")]
print(f"\n📁 Partitions created: {sorted(partitions)}")

# Read back to verify
verify = spark.read.parquet(f"{output_path}/banking_full")
print(f"✅ Verified read-back: {verify.count():,} rows")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("   ✅ PROJECT COMPLETE — SUMMARY")
print("=" * 60)
print(f"   Phase 1: Loaded 3 datasets (customers, accounts, transactions)")
print(f"   Phase 2: Cleaned dirty data — removed nulls, negatives, duplicates")
print(f"   Phase 3: Engineered features — time, amount category, spike flag")
print(f"   Phase 4: Joined 3 datasets into unified view")
print(f"   Phase 5: Aggregated spend per customer and category")
print(f"   Phase 6: Applied window functions (LAG, RANK, Running Total)")
print(f"   Phase 7: Compared DataFrame API vs Spark SQL — same results")
print(f"   Phase 8: Saved Parquet files partitioned by account_type")
print("=" * 60)

spark.stop()
