from inference import predict
import os

def monitor_batch(folder):
    total = 0
    defective = 0
    uncertain = 0

    for img in os.listdir(folder):
        result, confidence = predict(os.path.join(folder, img))
        total += 1

        if "DEFECTIVE" in result:
            defective += 1
        elif "UNCERTAIN" in result:
            uncertain += 1

    print("\n📊 BATCH QUALITY REPORT")
    print(f"Total items: {total}")
    print(f"Defective: {defective}")
    print(f"Uncertain: {uncertain}")
    print(f"Defect Rate: {defective/total:.2%}")

    if defective / total > 0.2:
        print("🚨 QUALITY ALERT: High defect rate detected")
