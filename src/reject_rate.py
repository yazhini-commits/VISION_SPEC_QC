import json

with open("../logs/predictions.json", "r") as f:
    logs = json.load(f)

total = len(logs)
rejected = sum(1 for l in logs if l["result"].startswith("UNCERTAIN"))

print("\nHUMAN-IN-THE-LOOP METRICS")
print(f"Total predictions: {total}")
print(f"Human review required: {rejected}")
print(f"Reject rate: {rejected/total:.2%}")
