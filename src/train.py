import os
import mlflow
import mlflow.data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


# --------------------------------------------------
# 1. Create a small sample fraud dataset
# --------------------------------------------------

df = pd.read_csv("data/transactions.csv")
dataset = mlflow.data.from_pandas(
    df,
    source="data/transactions.csv",
    name="fraud-transactions",
    targets="fraud"
)

mlflow.set_experiment("Fraud Detection")
# --------------------------------------------------
# 2. Convert country into a number
# --------------------------------------------------

encoder = LabelEncoder()

df["country"] = encoder.fit_transform(df["country"])


# --------------------------------------------------
# 3. Separate features and label
# --------------------------------------------------

X = df.drop("fraud", axis=1)
y = df["fraud"]


# --------------------------------------------------
# 4. Split data into training and testing
# --------------------------------------------------
n_estimators = 150
random_state = 32
test_size = 0.10
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)

with mlflow.start_run():

    # --------------------------------------------------
    # 5. Create the ML model
    # --------------------------------------------------
    
    mlflow.log_input(dataset, context="training")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("test_size", test_size)
    # --------------------------------------------------
    # 6. Train the model
    # --------------------------------------------------

    model.fit(X_train, y_train)

    # --------------------------------------------------
    # 7. Test the model
    # --------------------------------------------------

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", accuracy)

    model_info = mlflow.sklearn.log_model(
        model,
        name = "fraud_model"
    )

    mlflow.register_model(
    model_uri=model_info.model_uri,
    name="FraudDetectionModel"
    )
    print("\n===== Fraud Detection Model v1 =====")
    print(f"Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # --------------------------------------------------
    # 8. Save the trained model
    # --------------------------------------------------

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "encoder": encoder
        },
        "models/fraud_model_v1.pkl"
    )

    print("\nModel saved to:")
    print("models/fraud_model_v1.pkl")