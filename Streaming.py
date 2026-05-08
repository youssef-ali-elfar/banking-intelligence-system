"""
╔══════════════════════════════════════════════════════════════╗
║   ⚡ Streaming Data — Real-Time Transaction Simulation      ║
║   Using PySpark Structured Streaming                        ║
║   DS 405 - Big Data Analysis | Pharos University            ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import time
import json
import random
import threading
import warnings
warnings.filterwarnings("ignore")
os.environ["PYSPARK_PYTHON"] = "python3"

from faker import Faker
from datetime import datetime

print("=" * 60)
print("   ⚡ Real-Time Transaction Stream Simulator")
print("   DS 405 — Big Data Analysis")
print("=" * 60)

fake = Faker()
random.seed()

# ── Setup directories ──
STREAM_INPUT  = "data/streaming/input"
STREAM_OUTPUT = "data/streaming/output"
os.makedirs(STREAM_INPUT,  exist_ok=True)
os.makedirs(STREAM_OUTPUT, exist_ok=True)

ACCOUNT_IDS = [f"ACC_{i:05d}" for i in range(1, 50)]
TX_TYPES    = ["credit","debit","transfer","withdrawal","payment"]
CITIES      = ["Cairo","Alexandria","Giza","Aswan","Luxor","Mansoura"]
CATEGORIES  = ["groceries","electronics","travel","healthcare","entertainment","fuel"]

# ── Transaction Generator ──
def generate_transaction():
    amount = round(random.uniform(50, 55000), 2)
    is_suspicious = (amount > 40000 and random.random() < 0.4) or \
                    (random.random() < 0.05)
    return {
        "transaction_id": f"TXN_LIVE_{random.randint(100000,999999)}",
        "account_id":     random.choice(ACCOUNT_IDS),
        "amount":         amount,
        "tx_type":        random.choice(TX_TYPES),
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status":         random.choices(
            ["completed","pending","failed","reversed"],
            weights=[0.75, 0.1, 0.1, 0.05]
        )[0],
        "city":           random.choice(CITIES),
        "category":       random.choice(CATEGORIES),
        "is_suspicious":  int(is_suspicious)
    }

# ── Fraud Detection (simple rules engine) ──
def detect_fraud(tx):
    flags = []
    if tx["amount"] > 40000:
        flags.append("HIGH_AMOUNT")
    if tx["status"] in ["failed", "reversed"] and tx["amount"] > 10000:
        flags.append("FAILED_HIGH_TX")
    if tx["tx_type"] == "withdrawal" and tx["amount"] > 30000:
        flags.append("LARGE_WITHDRAWAL")
    return flags

# ── Simulate Spark Structured Streaming ──
def simulate_spark_streaming():
    """
    In a real environment with Kafka/socket source, you'd use:

    spark = SparkSession.builder.appName("BankingStream").getOrCreate()

    stream_df = spark.readStream \\
        .format("json") \\
        .schema(schema) \\
        .option("path", STREAM_INPUT) \\
        .load()

    fraud_stream = stream_df \\
        .filter(col("amount") > 40000) \\
        .withColumn("fraud_flag", lit("HIGH_AMOUNT"))

    query = fraud_stream.writeStream \\
        .outputMode("append") \\
        .format("console") \\
        .trigger(processingTime="5 seconds") \\
        .start()

    query.awaitTermination()

    Here we simulate the same logic with Python for demo purposes.
    """
    pass

# ── Main Streaming Loop ──
print("\n🚀 Starting live transaction stream...")
print("   Generating 1 transaction every 0.5 seconds for 20 seconds")
print("   Fraud detection running in real-time\n")
print(f"{'─'*65}")
print(f"{'Time':<10} {'TX ID':<22} {'Amount':>10} {'Type':<12} {'Status':<12} {'Alert'}")
print(f"{'─'*65}")

stats = {
    "total": 0, "fraud_flagged": 0,
    "total_amount": 0.0, "by_type": {},
    "alerts": []
}

batch_file_count = 0
batch_transactions = []

for i in range(40):  # 40 transactions over 20 seconds
    tx = generate_transaction()
    flags = detect_fraud(tx)
    tx["fraud_flags"] = flags
    tx["is_fraud"] = int(len(flags) > 0)

    # Update stats
    stats["total"] += 1
    stats["total_amount"] += tx["amount"]
    stats["by_type"][tx["tx_type"]] = stats["by_type"].get(tx["tx_type"], 0) + 1
    if flags:
        stats["fraud_flagged"] += 1
        stats["alerts"].append({
            "tx_id": tx["transaction_id"],
            "amount": tx["amount"],
            "flags": flags,
            "time": tx["timestamp"]
        })

    # Print to console (like Spark console sink)
    alert_str = f"⚠️  {','.join(flags)}" if flags else "✅ OK"
    time_str  = datetime.now().strftime("%H:%M:%S")
    amount_str= f"{tx['amount']:,.0f} EGP"
    print(f"{time_str:<10} {tx['transaction_id']:<22} {amount_str:>10} {tx['tx_type']:<12} {tx['status']:<12} {alert_str}")

    # Write to batch file (simulate micro-batch)
    batch_transactions.append(tx)
    if len(batch_transactions) >= 5:
        batch_file_count += 1
        batch_path = f"{STREAM_INPUT}/batch_{batch_file_count:04d}.json"
        with open(batch_path, "w") as f:
            for t in batch_transactions:
                f.write(json.dumps(t) + "\n")
        batch_transactions = []

    time.sleep(0.3)

# Write remaining
if batch_transactions:
    batch_file_count += 1
    with open(f"{STREAM_INPUT}/batch_{batch_file_count:04d}.json", "w") as f:
        for t in batch_transactions:
            f.write(json.dumps(t) + "\n")

print(f"{'─'*65}")

# ── Process all batches with PySpark ──
print("\n📊 Processing all batches with PySpark...")

import sys
os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("BankingStream") \
    .master("local[*]") \
    .config("spark.ui.showConsoleProgress","false") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

schema = StructType([
    StructField("transaction_id", StringType()),
    StructField("account_id",     StringType()),
    StructField("amount",         DoubleType()),
    StructField("tx_type",        StringType()),
    StructField("timestamp",      StringType()),
    StructField("status",         StringType()),
    StructField("city",           StringType()),
    StructField("category",       StringType()),
    StructField("is_suspicious",  IntegerType()),
    StructField("is_fraud",       IntegerType()),
])

# Read all batch files (simulating readStream result)
batch_df = spark.read.json(STREAM_INPUT, schema=schema)

print(f"\n✅ Total streamed transactions processed: {batch_df.count()}")

print("\n📈 Real-Time Aggregations:")
print("→ Fraud rate by city:")
batch_df.groupBy("city").agg(
    F.count("*").alias("total"),
    F.sum("is_fraud").alias("fraud_count"),
    F.round(F.mean("is_fraud") * 100, 1).alias("fraud_rate_%")
).orderBy(F.desc("fraud_rate_%")).show()

print("→ Volume by transaction type:")
batch_df.groupBy("tx_type").agg(
    F.count("*").alias("count"),
    F.round(F.sum("amount"), 2).alias("total_amount")
).orderBy(F.desc("total_amount")).show()

print("→ Flagged transactions:")
batch_df.filter(F.col("is_fraud") == 1) \
    .select("transaction_id","city","amount","tx_type","status") \
    .show(10, truncate=False)

# Save results
batch_df.write.mode("overwrite").parquet(STREAM_OUTPUT)
print(f"✅ Stream results saved → {STREAM_OUTPUT}/")

spark.stop()

# ── Final Summary ──
fraud_rate = stats["fraud_flagged"] / stats["total"] * 100
print("\n" + "=" * 60)
print("   ⚡ STREAMING SIMULATION COMPLETE!")
print("=" * 60)
print(f"   Transactions Processed: {stats['total']}")
print(f"   Total Volume:           {stats['total_amount']:,.0f} EGP")
print(f"   Fraud Flagged:          {stats['fraud_flagged']} ({fraud_rate:.1f}%)")
print(f"   Batch Files Written:    {batch_file_count}")
print(f"\n   🔴 LIVE ALERTS:")
for alert in stats["alerts"][:5]:
    print(f"   ⚠️  {alert['tx_id']} | {alert['amount']:,.0f} EGP | {', '.join(alert['flags'])}")
print("=" * 60)
print("""
   📝 NOTE: In production, replace the file-based input with:

   stream_df = spark.readStream \\
       .format("kafka") \\
       .option("kafka.bootstrap.servers", "localhost:9092") \\
       .option("subscribe", "banking_transactions") \\
       .load()

   Or use socket source for demo:
   stream_df = spark.readStream \\
       .format("socket") \\
       .option("host", "localhost") \\
       .option("port", 9999) \\
       .load()
""")