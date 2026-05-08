"""
Phase 1: Data Generation
Banking Transactions Intelligence System
DS 405 - Big Data Analysis
"""

import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
random.seed(42)
Faker.seed(42)

# ──────────────────────────────────────────
# 1. Generate Customers
# ──────────────────────────────────────────
def generate_customers(n=200):
    cities = ["Cairo", "Alexandria", "Giza", "Luxor", "Aswan", "Mansoura"]
    records = []
    for i in range(1, n + 1):
        records.append({
            "customer_id": f"CUST_{i:04d}",
            "name": fake.name(),
            "age": random.randint(18, 75),
            "city": random.choice(cities),
            "join_date": fake.date_between(start_date="-5y", end_date="today").strftime("%Y-%m-%d"),
            "email": fake.email(),
            "phone": fake.phone_number()[:15]
        })
    df = pd.DataFrame(records)
    # Inject some nulls for cleaning
    df.loc[df.sample(frac=0.03).index, "age"] = None
    df.loc[df.sample(frac=0.02).index, "city"] = None
    return df


# ──────────────────────────────────────────
# 2. Generate Accounts
# ──────────────────────────────────────────
def generate_accounts(customers_df):
    account_types = ["savings", "current", "fixed_deposit", "investment"]
    records = []
    account_id = 1
    for _, cust in customers_df.iterrows():
        num_accounts = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        for _ in range(num_accounts):
            records.append({
                "account_id": f"ACC_{account_id:05d}",
                "customer_id": cust["customer_id"],
                "account_type": random.choice(account_types),
                "balance": round(random.uniform(100, 500000), 2),
                "open_date": fake.date_between(start_date="-4y", end_date="today").strftime("%Y-%m-%d"),
                "status": random.choices(["active", "inactive", "frozen"], weights=[0.85, 0.1, 0.05])[0]
            })
            account_id += 1
    df = pd.DataFrame(records)
    # Inject some nulls
    df.loc[df.sample(frac=0.02).index, "balance"] = None
    return df


# ──────────────────────────────────────────
# 3. Generate Transactions
# ──────────────────────────────────────────
def generate_transactions(accounts_df, n=2000):
    tx_types = ["credit", "debit", "transfer", "withdrawal", "payment"]
    statuses = ["completed", "pending", "failed", "reversed"]
    records = []
    account_ids = accounts_df["account_id"].tolist()

    base_date = datetime(2024, 1, 1)

    for i in range(1, n + 1):
        tx_date = base_date + timedelta(days=random.randint(0, 365))
        amount = round(random.uniform(10, 50000), 2)

        records.append({
            "transaction_id": f"TXN_{i:06d}",
            "account_id": random.choice(account_ids),
            "amount": amount,
            "tx_type": random.choice(tx_types),
            "timestamp": tx_date.strftime("%Y-%m-%d %H:%M:%S"),
            "status": random.choices(statuses, weights=[0.75, 0.1, 0.1, 0.05])[0],
            "merchant": fake.company(),
            "category": random.choice(["groceries", "electronics", "travel", "healthcare",
                                        "entertainment", "utilities", "restaurant", "fuel"])
        })

    df = pd.DataFrame(records)

    # ── Inject dirty data for cleaning phase ──
    # Negative amounts
    df.loc[df.sample(frac=0.03).index, "amount"] = -abs(df["amount"])
    # Null amounts
    df.loc[df.sample(frac=0.02).index, "amount"] = None
    # Null tx_type
    df.loc[df.sample(frac=0.02).index, "tx_type"] = None
    # Duplicates
    duplicates = df.sample(n=20)
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)

    print("⏳ Generating customers...")
    customers = generate_customers(200)
    customers.to_csv("data/raw/customers.csv", index=False)
    print(f"✅ Customers: {len(customers)} records → data/raw/customers.csv")

    print("⏳ Generating accounts...")
    accounts = generate_accounts(customers)
    accounts.to_csv("data/raw/accounts.csv", index=False)
    print(f"✅ Accounts:  {len(accounts)} records → data/raw/accounts.csv")

    print("⏳ Generating transactions...")
    transactions = generate_transactions(accounts, n=2000)
    transactions.to_csv("data/raw/transactions.csv", index=False)
    print(f"✅ Transactions: {len(transactions)} records → data/raw/transactions.csv")

    print("\n📊 Sample dirty stats:")
    print(f"   Null amounts:     {transactions['amount'].isna().sum()}")
    print(f"   Negative amounts: {(transactions['amount'] < 0).sum()}")
    print(f"   Null tx_type:     {transactions['tx_type'].isna().sum()}")
    print(f"   Duplicate TXN IDs:{transactions.duplicated('transaction_id').sum()}")
    print("\n🎉 Data generation complete!")
