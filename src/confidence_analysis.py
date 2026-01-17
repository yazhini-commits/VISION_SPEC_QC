import json
import matplotlib.pyplot as plt

with open("../logs/predictions.json", "r") as f:
    logs = json.load(f)

confidences = [entry["confidence"] for entry in logs]

plt.hist(confidences, bins=10)
plt.title("Model Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.show()
