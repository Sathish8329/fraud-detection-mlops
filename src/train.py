import os
import argparse

import mlflow
import mlflow.data
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import joblib


# --------------------------------------------------
# 1. Read command-line parameters
# --------------------------------------------------

parser = argparse.ArgumentParser(description="Fraud Detection Training")

parser.add_argument(
    "--n-estimators",
    type=int,
    default=300
)

parser.add_argument(
    "--random-state",
    type=int,
    default=42
)

parser.add_argument(
    "--test-size",
    type=float,
    default=0.30
)

args = parser.parse_args()

n_estimators = args.n_estimators
random_state = args.random_state
test_size = args.test_size


# --------------------------------------------------
# 2. Load dataset
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
# 3. Convert country into a number
# --------------------------------------------------

encoder = LabelEncoder()

df["country"] = encoder.fit_transform(df["country"])


# --------------------------------------------------
# 4. Separate features and label
# --------------------------------------------------

X = df.drop("fraud", axis=1)
y = df["fraud"]


# --------------------------------------------------
# 5. Split data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)


# --------------------------------------------------
# 6. Start MLflow run
# --------------------------------------------------

with mlflow.start_run():

    mlflow.log_input(
        dataset,
        context="training"
    )

    # --------------------------------------------------
    # 7. Create model
    # --------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    # --------------------------------------------------
    # 8. Log parameters
    # --------------------------------------------------

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "random_state",
        random_state
    )

    mlflow.log_param(
        "test_size",
        test_size
    )

    # --------------------------------------------------
    # 9. Train model
    # --------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # 10. Prediction
    # --------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------
    # 11. Evaluation
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    # --------------------------------------------------
    # 12. Log model to MLflow
    # --------------------------------------------------

    model_info = mlflow.sklearn.log_model(
        model,
        name="fraud_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    # --------------------------------------------------
    # 13. Register model
    # --------------------------------------------------

    # mlflow.register_model(
    #     model_uri=model_info.model_uri,
    #     name="FraudDetectionModel"
    # )

    # --------------------------------------------------
# 13. Register model
# --------------------------------------------------

    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name="FraudDetectionModel"
    )
    run_id = mlflow.active_run().info.run_id

    model_version = registered_model.version

    training_result = {
      "run_id": run_id,
      "accuracy": accuracy,
      "model_version": model_version
    }

    os.makedirs("models", exist_ok=True)

    with open("models/training_result.json", "w") as f:
        import json
        json.dump(training_result, f, indent=2)

    print("\nTraining result saved to:")
    print("models/training_result.json")

    # --------------------------------------------------
    # 14. Print results
    # --------------------------------------------------

    print("\n===================================")
    print("Fraud Detection Training")
    print("===================================")

    print(f"n_estimators : {n_estimators}")
    print(f"random_state : {random_state}")
    print(f"test_size    : {test_size}")
    print(f"Accuracy     : {accuracy:.2f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    # --------------------------------------------------
    # 15. Save local model
    # --------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "encoder": encoder
        },
        "models/fraud_model.pkl"
    )

    print("\nModel saved to:")
    print("models/fraud_model.pkl")

