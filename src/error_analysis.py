import json
import os

LOG_FILE = "../logs/predictions.json"

if not os.path.exists(LOG_FILE):
    print("No prediction logs found")
    exit()

with open(LOG_FILE, "r") as f:
    logs = json.load(f)

low_conf = 0
drift = 0
high_sev = 0

for entry in logs:
    if entry["result"].startswith("UNCERTAIN"):
        low_conf += 1
    if entry["data_drift_warning"]:
        drift += 1
    if entry["severity"] == "HIGH":
        high_sev += 1

print("\nERROR ANALYSIS SUMMARY")
print(f"Uncertain predictions: {low_conf}")
print(f"Drift warnings: {drift}")
print(f"High severity defects: {high_sev}")
