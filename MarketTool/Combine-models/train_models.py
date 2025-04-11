# train_live_models.py
import os
import pandas as pd
import joblib
from data_prep import download_stock_data, create_labels, get_sp500_tickers
from indicators import generate_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

def build_dataset_from_live_data(limit=50, start="2018-01-01", end="2024-12-31"):
    tickers = get_sp500_tickers()[:limit]
    all_data = []

    for i, ticker in enumerate(tickers):
        try:
            print(f"[{i+1}/{limit}] Fetching {ticker}...")
            df = download_stock_data(ticker, start, end)

            if df.empty or len(df) < 100:
                print(f"Skipping {ticker} (not enough data)")
                continue

            features = generate_features(df)
            labels = create_labels(df)

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

def train_models_from_live_data():
    df = build_dataset_from_live_data(limit=50)
    if df.empty:
        print("No data to train on.")
        return

    X = df.drop(columns=["Label", "Ticker", "Date"], errors="ignore")
    y = df["Label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # === XGBoost ===
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    print("\nXGBoost:")
    print(classification_report(y_test, xgb_preds, target_names=encoder.classes_))

    # === Grid Search for XGBoost ===
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1],
    }
    xgb_grid = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
        param_grid=xgb_params,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    xgb_grid.fit(X_train, y_train)
    xgb = xgb_grid.best_estimator_
    xgb_preds = xgb.predict(X_test)
    print("\nXGBoost Best Params:", xgb_grid.best_params_)
    print(classification_report(y_test, xgb_preds, target_names=encoder.classes_))


    # === Logistic Regression ===
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train)
    logreg_preds = logreg.predict(X_test)
    print("\nLogistic Regression:")
    print(classification_report(y_test, logreg_preds, target_names=encoder.classes_))

    # === Grid Search for Logistic Regression ===
    logreg_params = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'saga'],
        'penalty': ['l2'],
    }
    logreg_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid=logreg_params,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    logreg_grid.fit(X_train, y_train)
    logreg = logreg_grid.best_estimator_
    logreg_preds = logreg.predict(X_test)
    print("\nLogistic Regression Best Params:", logreg_grid.best_params_)
    print(classification_report(y_test, logreg_preds, target_names=encoder.classes_))

    # === MLP Classifier ===
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_preds = mlp.predict(X_test)
    print("\nMLP Classifier:")
    print(classification_report(y_test, mlp_preds, target_names=encoder.classes_))

    # === Grid Search for MLPClassifier ===
    mlp_params = {
        'hidden_layer_sizes': [(64,), (100,), (150,)],
        'alpha': [0.0001, 0.001, 0.01],  # L2 regularisation
        'learning_rate_init': [0.001, 0.01],
    }
    mlp_grid = GridSearchCV(
        MLPClassifier(max_iter=1000, random_state=42),
        param_grid=mlp_params,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    mlp_grid.fit(X_train, y_train)
    mlp = mlp_grid.best_estimator_
    mlp_preds = mlp.predict(X_test)
    print("\nMLP Classifier Best Params:", mlp_grid.best_params_)
    print(classification_report(y_test, mlp_preds, target_names=encoder.classes_))

    # === Voting Classifier (XGB + LogReg + MLP) ===
    voting_clf = VotingClassifier(estimators=[
        ('xgb', xgb),
        ('logreg', logreg),
        ('mlp', mlp)
    ], voting='soft')

    voting_clf.fit(X_train, y_train)
    voting_preds = voting_clf.predict(X_test)
    print("\nVoting Classifier (XGB + LogReg + MLP):")
    print(classification_report(y_test, voting_preds, target_names=encoder.classes_))

    # Save models
    os.makedirs("MarketTool/Combine-models/models/", exist_ok=True)
    joblib.dump(xgb, "MarketTool/Combine-models/models/xgb_model.pkl")
    joblib.dump(logreg, "MarketTool/Combine-models/models/logreg_model.pkl")
    joblib.dump(mlp, "MarketTool/Combine-models/models/mlp_model.pkl")
    joblib.dump(voting_clf, "MarketTool/Combine-models/models/voting_classifier.pkl")
    joblib.dump(scaler, "MarketTool/Combine-models/models/scaler.pkl")
    joblib.dump(encoder, "MarketTool/Combine-models/models/label_encoder.pkl")
    print("\nModels saved to /models/")

if __name__ == "__main__":
    train_models_from_live_data()