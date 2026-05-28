import os
import matplotlib.pyplot as plt
import pandas as pd
from data.market_data import fetch_crypto_data
from data.sentiment import fetch_news_sentiment
from models.predictor import prepare_data, train_and_predict

def run_pipeline():
    print("Fetching Market Data...")
    market_df = fetch_crypto_data("BTC-USD", period='2y')
    
    print("Fetching Sentiment Data...")
    sentiment_df = fetch_news_sentiment("Bitcoin", days=30)
    
    print("Preparing Data...")
    X, y, X_predict, features, final_df = prepare_data(market_df, sentiment_df)
    
    print("Training Models...")
    results, best_name, importance_df = train_and_predict(X, y, X_predict, features)
    
    print(f"Best Model: {best_name} (Accuracy: {results[best_name]['Accuracy']:.2f})")
    
    # Save Outputs
    print("Saving Outputs...")
    final_df.to_csv("final_merged_data.csv")
    
    # Save Model predictions
    preds_data = []
    for name, res in results.items():
        preds_data.append({
            'Model': name,
            'Accuracy': res['Accuracy'],
            'Prediction': res['Prediction'],
            'Confidence': res['Confidence']
        })
    pd.DataFrame(preds_data).to_csv("model_predictions.csv", index=False)
    
    # Save Feature Importance Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
    plt.title(f"Feature Importance ({best_name})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    
    print("Pipeline Complete! Files saved: final_merged_data.csv, model_predictions.csv, feature_importance.png")

if __name__ == "__main__":
    run_pipeline()
