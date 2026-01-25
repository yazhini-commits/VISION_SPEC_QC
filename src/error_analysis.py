import json
import os

# --------------------------------------------------
# PATH SETUP (ROBUST & CORRECT)
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "predictions.json")

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------
if not os.path.exists(LOG_FILE):
    print("No prediction logs found.")
    print(f"Expected at: {LOG_FILE}")
    print("Run inference or demo script first to generate logs.")
    exit()

# --------------------------------------------------
# LOAD LOGS
# --------------------------------------------------
with open(LOG_FILE, "r") as f:
    logs = json.load(f)

# --------------------------------------------------
# METRICS INITIALIZATION
# --------------------------------------------------
total = len(logs)

uncertain = 0
drift_warnings = 0
high_severity = 0

false_negative = 0   # DEFECTIVE marked as GOOD (dangerous)
false_positive = 0   # GOOD marked as DEFECTIVE (safe but costly)

# --------------------------------------------------
# ANALYSIS LOOP
# --------------------------------------------------
for entry in logs:
    result = entry["result"]
    severity = entry["severity"]
    drift = entry["data_drift_warning"]

    if result.startswith("UNCERTAIN"):
        uncertain += 1

    if drift:
        drift_warnings += 1

    if severity == "HIGH":
        high_severity += 1

    # QC-critical errors
    if result == "GOOD" and severity == "HIGH":
        false_negative += 1

    if result == "DEFECTIVE" and severity is None:
        false_positive += 1

# --------------------------------------------------
# REPORT
# --------------------------------------------------
print("\nQUALITY CONTROL ERROR ANALYSIS (DAY 17)")
print("-" * 45)

print(f"Total predictions            : {total}")
print(f"Uncertain (human review)     : {uncertain}")
print(f"Drift warnings               : {drift_warnings}")
print(f"High severity defects        : {high_severity}")

print("\nQC-CRITICAL METRICS")
print(f"Critical false negatives     : {false_negative}")
print(f"False positives (safe cost)  : {false_positive}")

if false_negative > 0:
    print("\nALERT: Unsafe predictions detected.")
else:
    print("\nNo critical false negatives detected.")

print("\nPolicy: Reject-first QC (false positives preferred over false negatives)")
