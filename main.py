import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/churn_data.csv")

print("Dataset Preview:")
print(df.head())

# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop missing values
df.dropna(inplace=True)

# Encode categorical columns
encoder = LabelEncoder()

for col in df.select_dtypes(include=['object', 'string']).columns:
    if col != "Churn":
        df[col] = encoder.fit_transform(df[col])

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Simple plot
df["Churn"].value_counts().plot(kind="bar")
plt.title("Churn Distribution")
plt.show()