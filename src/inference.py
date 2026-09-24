import joblib
import pandas as pd


# Load trained model
model_data = joblib.load("models/fraud_model.pkl")

model = model_data["model"]
encoder = model_data["encoder"]


# Get transaction details from user
amount = float(input("Enter transaction amount: "))
country = input("Enter country code: ").strip().upper()
hour = int(input("Enter transaction hour (0-23): "))
previous_txns = int(input("Enter number of previous transactions: "))


# Create transaction
transaction = pd.DataFrame([
    {
        "amount": amount,
        "country": country,
        "hour": hour,
        "previous_txns": previous_txns
    }
])


# Encode country
transaction["country"] = encoder.transform(
    transaction["country"]
)


# Make prediction
prediction = model.predict(transaction)


# Display result
if prediction[0] == 1:
    print("\n🚨 FRAUD TRANSACTION")
else:
    print("\n✅ LEGITIMATE TRANSACTION")