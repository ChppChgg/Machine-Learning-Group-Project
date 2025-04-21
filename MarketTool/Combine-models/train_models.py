import os
import pandas as pd
import joblib
from data_prep import download_stock_data, create_labels, get_sp500_tickers
from indicators import generate_features
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# === STEP 1: Build the training dataset ===
def build_dataset_from_live_data(limit=50, start="2021-01-01", end="2024-01-05"):
    tickers = get_sp500_tickers()[:limit]  # Limit the number of S&P 500 stocks
    all_data = []

    for i, ticker in enumerate(tickers):
        try:
            print(f"[{i+1}/{limit}] Fetching {ticker}...")
            df = download_stock_data(ticker, start, end)
            if df.empty or len(df) < 100:
                print(f"Skipping {ticker} (not enough data)")
                continue

            # Generate indicators + sentiment features
            features = generate_features(df, ticker)

            # Generate buy/sell/hold labels
            labels = create_labels(df)

            # Make sure features and labels align by date
            common_index = features.index.intersection(labels.index)
            features = features.loc[common_index].copy()
            labels = labels.loc[common_index]

            if features.empty or len(features) < 10:
                print(f"Skipping {ticker} (not enough aligned rows)")
                continue

            features["Label"] = labels
            features["Ticker"] = ticker
            features.dropna(inplace=True)
            all_data.append(features)

        except Exception as e:
            print(f"Failed to process {ticker}: {e}")

    return pd.concat(all_data) if all_data else pd.DataFrame()

# === STEP 2: Train models ===
def train_models_from_live_data():
    # Build dataset from historical data
    df = build_dataset_from_live_data(limit=500, start="2023-01-01", end="2025-01-01")

    # Only keep BUY and SELL samples (remove HOLD)
    df = df[df["Label"].isin(["BUY", "SELL"])]
    if df.empty:
        print("No data to train on.")
        return

    # === Data Preparation ===
    X = df.drop(columns=["Label", "Ticker", "Date"], errors="ignore")
    y = df["Label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Handle class imbalance with SMOTE (upsampling)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # === Train XGBoost with hyperparameter tuning ===
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    xgb_grid = GridSearchCV(
        XGBClassifier(eval_metric="mlogloss", random_state=42),
        param_grid=xgb_params,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    xgb_grid.fit(X_train, y_train)
    xgb = xgb_grid.best_estimator_
    print("\nXGBoost Best Params:", xgb_grid.best_params_)

    # === Train Logistic Regression with tuning ===
    logreg_params = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'saga'],
        'penalty': ['l2']
    }
    logreg_grid = GridSearchCV(
        LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42),
        param_grid=logreg_params,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    logreg_grid.fit(X_train, y_train)
    logreg = logreg_grid.best_estimator_
    print("\nLogistic Regression Best Params:", logreg_grid.best_params_)

    # === Train MLP Neural Network with tuning ===
    mlp_params = {
        'hidden_layer_sizes': [(64,), (100,), (150,)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
        'activation': ['relu'],
        'solver': ['adam']
    }
    mlp_grid = GridSearchCV(
        MLPClassifier(max_iter=2000, random_state=42),
        param_grid=mlp_params,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    mlp_grid.fit(X_train, y_train)
    mlp = mlp_grid.best_estimator_
    print("\nMLP Classifier Best Params:", mlp_grid.best_params_)

    # === Combine all 3 models into a stacking ensemble ===
    stacked_clf = StackingClassifier(
        estimators=[
            ('xgb', xgb),
            ('logreg', logreg),
            ('mlp', mlp)
        ],
        final_estimator=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        passthrough=True,
        n_jobs=-1
    )
    stacked_clf.fit(X_train, y_train)

    # === Evaluate model performance ===
    y_pred = stacked_clf.predict(X_test)
    print("\nStacked Classifier Evaluation (BUY vs SELL only):")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # === Save models and encoders ===
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb, "MarketTool/Combine-models/models/xgb_model.pkl")
    joblib.dump(logreg, "MarketTool/Combine-models/models/logreg_model.pkl")
    joblib.dump(mlp, "MarketTool/Combine-models/models/mlp_model.pkl")
    joblib.dump(stacked_clf, "MarketTool/Combine-models/models/stacked_classifier.pkl")
    joblib.dump(scaler, "MarketTool/Combine-models/models/scaler.pkl")
    joblib.dump(encoder, "MarketTool/Combine-models/models/label_encoder.pkl")
    print("\nModels saved to /models/")

# === Entry Point ===
if __name__ == "__main__":
    train_models_from_live_data()