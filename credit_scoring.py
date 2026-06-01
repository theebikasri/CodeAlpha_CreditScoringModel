import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("credit_risk_dataset.csv")

# Remove missing values
data = data.dropna()

# Encode categorical columns
le = LabelEncoder()

data["person_home_ownership"] = le.fit_transform(data["person_home_ownership"])
data["loan_intent"] = le.fit_transform(data["loan_intent"])
data["loan_grade"] = le.fit_transform(data["loan_grade"])
data["cb_person_default_on_file"] = le.fit_transform(data["cb_person_default_on_file"])

# Features
X = data.drop("loan_status", axis=1)

# Target
y = data["loan_status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")