import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("new/csv_files/training_data.csv")

# Drop unused columns
X = df.drop(columns=["Date", "Ticker", "Label"])
y = df["Label"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data (even if small, needed for structure)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Train XGBoost
xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)
xgb.fit(X_train, y_train)

# Train Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Create model directory if needed
os.makedirs("new/models", exist_ok=True)

# Save everything
joblib.dump(rf, "new/models/random_forest_model.pkl")
joblib.dump(xgb, "new/models/xgboost_model.pkl")
joblib.dump(lr, "new/models/logistic_model.pkl")
joblib.dump(scaler, "new/models/scaler.pkl")
joblib.dump(label_encoder, "new/models/label_encoder.pkl")

print("Models trained and saved successfully.")
