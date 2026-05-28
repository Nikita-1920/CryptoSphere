import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def prepare_data(market_df, sentiment_df):
    """Merge data and create features/target."""
    df = market_df.copy()
    df.index = df.index.tz_localize(None).normalize()
    
    if not sentiment_df.empty:
        sentiment_df.index = pd.to_datetime(sentiment_df.index).normalize()
        df = df.join(sentiment_df[['Sentiment']], how='left')
        df['Sentiment'] = df['Sentiment'].fillna(method='ffill').fillna(0)
    else:
        df['Sentiment'] = 0.0
        
    # Target: 1 if next day's close > today's close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    features = ['Close', 'Volume', 'MA_7', 'MA_14', 'RSI_14', 'Volatility', 'Momentum', 'Lag_1', 'Sentiment']
    df = df.dropna(subset=features)
    
    predict_df = df.iloc[-1:]
    train_df = df.iloc[:-1]
    
    X = train_df[features]
    y = train_df['Target']
    X_predict = predict_df[features]
    
    return X, y, X_predict, features, df

def train_and_predict(X, y, X_predict, features):
    """Train multiple models and compare them."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    best_model = None
    best_acc = 0
    best_name = ""
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        prob = model.predict_proba(X_predict)[0]
        prediction = "Up" if prob[1] > 0.5 else "Down"
        confidence = prob[1] if prob[1] > 0.5 else prob[0]
        
        results[name] = {
            'Accuracy': acc,
            'Prediction': prediction,
            'Confidence': confidence,
            'Model': model,
            'Classification_Report': classification_report(y_test, preds, output_dict=True, zero_division=0),
            'Confusion_Matrix': confusion_matrix(y_test, preds).tolist()
        }
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        importances = models['Random Forest'].feature_importances_

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return results, best_name, importance_df

def predict_future_7_days(df, best_model, features):
    """Iterative 7-day prediction using the trained model."""
    last_row = df.iloc[-1].copy()
    predictions = []
    
    for i in range(7):
        X_pred = pd.DataFrame([last_row[features]])
        prob = best_model.predict_proba(X_pred)[0]
        pred_dir = 1 if prob[1] > 0.5 else 0
        
        pct_change = 0.02 if pred_dir == 1 else -0.02
        new_close = last_row['Close'] * (1 + pct_change)
        
        predictions.append({
            'Day': f"Day {i+1}",
            'Predicted_Close': new_close,
            'Direction': 'Up' if pred_dir == 1 else 'Down',
            'Confidence': prob[1] if pred_dir == 1 else prob[0]
        })
        
        last_row['Lag_1'] = last_row['Close']
        last_row['Close'] = new_close
        
    return pd.DataFrame(predictions)
