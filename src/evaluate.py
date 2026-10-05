import json
import sys

THRESHOLD = 0.80

with open("models/training_result.json") as f:
    result = json.load(f)

accuracy = result["accuracy"]

print(f"Model accuracy: {accuracy:.4f}")
print(f"Minimum required: {THRESHOLD:.4f}")

if accuracy < THRESHOLD:
    print("❌ Model quality gate failed.")
    sys.exit(1)

print("✅ Model quality gate passed.")
