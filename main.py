"""
╔══════════════════════════════════════════════════════════════╗
║   ▶️  MAIN — Run Full Project                               ║
║   Banking Transactions Intelligence System                   ║
║   DS 405 - Big Data Analysis | Pharos University            ║
╚══════════════════════════════════════════════════════════════╝

Run this file to execute all phases in order:
  1. Data Generation
  2. PySpark Analytics Pipeline
  3. AI Fraud Detection Model (BONUS)
  4. Visualizations Dashboard
"""

import subprocess
import sys
import time
import io
import os

# Force UTF-8 output on Windows so emoji characters print correctly
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"

def run_step(script, title):
    print(f"\n{'='*60}")
    print(f"  ▶️  {title}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n  ✅ Done in {elapsed:.1f}s")
    else:
        print(f"\n  ❌ Error in {script}")
        sys.exit(1)

if __name__ == "__main__":
    print("\n" + "🏦 " * 20)
    print("  BANKING TRANSACTIONS INTELLIGENCE SYSTEM")
    print("  DS 405 — Big Data Analysis | Pharos University")
    print("🏦 " * 20)

    total_start = time.time()

    run_step("generate_data.py",     "Phase 1 — Generating Synthetic Data")
    run_step("banking_analytics.py", "Phases 2-8 — PySpark Analytics Pipeline")
    run_step("ai_model.py",          "BONUS — AI Fraud Detection Model")
    run_step("visualize.py",         "Phase 9 — Dashboard Visualizations")

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  🎉 ALL PHASES COMPLETE in {total:.1f}s!")
    print(f"{'='*60}")
    print(f"\n  📁 Output files:")
    print(f"     data/raw/           → CSV datasets")
    print(f"     data/processed/     → Parquet (partitioned)")
    print(f"     data/banking_dashboard.png      → Analytics Dashboard")
    print(f"     data/fraud_detection_report.png → AI Model Report")
    print(f"\n  ✅ Project ready for submission!")
    print(f"{'='*60}\n")
