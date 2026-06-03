import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Load dataset
data = pd.read_csv("credit_risk_dataset.csv")

# Remove missing values
data = data.dropna()

# Encode categorical columns
le = LabelEncoder()

data["person_home_ownership"] = le.fit_transform(
    data["person_home_ownership"]
)

data["loan_intent"] = le.fit_transform(
    data["loan_intent"]
)

data["loan_grade"] = le.fit_transform(
    data["loan_grade"]
)

data["cb_person_default_on_file"] = le.fit_transform(
    data["cb_person_default_on_file"]
)

# Features
X = data.drop("loan_status", axis=1)

# Target
y = data["loan_status"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Print Results
print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\n===== CONFUSION MATRIX =====")
print(cm)

# Save results to file
with open("results.txt", "w") as file:
    file.write("CREDIT RISK PREDICTION MODEL RESULTS\n")
    file.write("-----------------------------------\n")
    file.write(f"Accuracy  : {accuracy:.4f}\n")
    file.write(f"Precision : {precision:.4f}\n")
    file.write(f"Recall    : {recall:.4f}\n")
    file.write(f"F1 Score  : {f1:.4f}\n")
    file.write(f"ROC-AUC   : {roc_auc:.4f}\n\n")
    file.write("Confusion Matrix:\n")
    file.write(str(cm))

# Save trained model
pickle.dump(model, open("model.pkl", "wb"))

print("\nModel saved successfully as model.pkl")
print("Results saved successfully as results.txt")