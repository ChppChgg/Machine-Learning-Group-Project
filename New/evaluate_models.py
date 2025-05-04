import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("new/csv_files/training_data.csv")
X = df.drop(columns=["Date", "Ticker", "Label"])
y = df["Label"]

# Load scaler and encoder
scaler = joblib.load("new/models/scaler.pkl")
label_encoder = joblib.load("new/models/label_encoder.pkl")

# Encode labels and scale features
y_encoded = label_encoder.transform(y)
X_scaled = scaler.transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Load models
rf = joblib.load("new/models/random_forest_model.pkl")
xgb = joblib.load("new/models/xgboost_model.pkl")
lr = joblib.load("new/models/logistic_model.pkl")

# Evaluation function
def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    print(f"\n{name} Evaluation:")
    print("----------------------------")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
    print("F1 Score :", f1_score(y_test, y_pred, average="weighted"))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(
        y_test, y_pred, target_names=label_encoder.classes_))

# Run evaluations
evaluate_model(rf, "Random Forest")
evaluate_model(xgb, "XGBoost")
evaluate_model(lr, "Logistic Regression")
