import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import pandas as pd
from datetime import datetime
import nltk
from nltk.corpus import stopwords
import re

nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def clean_text(text):
    """Clean text by removing links, special characters, and stopwords."""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    words = text.lower().split()
    try:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    except:
        pass
    return ' '.join(words)

def fetch_news_sentiment(query="cryptocurrency", days=30):
    """Fetch and analyze news sentiment."""
    try:
        url = f"https://news.google.com/rss/search?q={query}+when:{days}d&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, features="xml")
        articles = soup.findAll("item")
        
        analyzer = SentimentIntensityAnalyzer()
        data = []
        for article in articles:
            raw_title = article.title.text
            clean_title = clean_text(raw_title)
            pubDate = article.pubDate.text
            
            try:
                date_obj = datetime.strptime(pubDate, "%a, %d %b %Y %H:%M:%S %Z").date()
            except:
                date_obj = datetime.now().date()
            
            blob = TextBlob(clean_title)
            tb_polarity = blob.sentiment.polarity
            vader_scores = analyzer.polarity_scores(clean_title)
            compound = vader_scores['compound']
            
            avg_sentiment = (tb_polarity + compound) / 2.0
            
            data.append({
                'Date': pd.to_datetime(date_obj),
                'Title': raw_title,
                'Cleaned_Text': clean_title,
                'Sentiment_Score': avg_sentiment,
                'Sentiment_Label': 1 if avg_sentiment > 0.05 else (-1 if avg_sentiment < -0.05 else 0)
            })
            
        df = pd.DataFrame(data)
        if not df.empty:
            daily_sentiment = df.groupby('Date').agg(
                Sentiment=('Sentiment_Score', 'mean'),
                Sentiment_Count=('Title', 'count')
            ).reset_index()
            daily_sentiment.set_index('Date', inplace=True)
            return daily_sentiment
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching sentiment: {e}")
        return pd.DataFrame()

def fetch_raw_news(query="cryptocurrency", days=7):
    """Fetch unaggregated news articles for the News Feed tab."""
    try:
        url = f"https://news.google.com/rss/search?q={query}+when:{days}d&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, features="xml")
        articles = soup.findAll("item")
        
        analyzer = SentimentIntensityAnalyzer()
        data = []
        for article in articles:
            raw_title = article.title.text
            clean_title = clean_text(raw_title)
            link = article.link.text
            pubDate = article.pubDate.text
            
            blob = TextBlob(clean_title)
            tb_polarity = blob.sentiment.polarity
            vader_scores = analyzer.polarity_scores(clean_title)
            compound = vader_scores['compound']
            
            avg_sentiment = (tb_polarity + compound) / 2.0
            
            data.append({
                'Date': pubDate,
                'Title': raw_title,
                'Link': link,
                'Sentiment_Score': avg_sentiment,
            })
            
        df = pd.DataFrame(data)
        if not df.empty:
            # Parse dates and sort descending (newest first)
            df['Parsed_Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values(by='Parsed_Date', ascending=False)
            
            # Format the date cleanly for display
            df['Date'] = df['Parsed_Date'].dt.strftime('%b %d, %Y %I:%M %p')
            df = df.drop(columns=['Parsed_Date'])
            
            return df.head(20)
        return df
    except Exception as e:
        print(f"Error fetching raw news: {e}")
        return pd.DataFrame()

def fetch_reddit_sentiment(query="Bitcoin", limit=20):
    """
    Scrape recent posts from r/CryptoCurrency mentioning the query and run VADER sentiment.
    Returns a normalized Fear & Greed score (0 to 100).
    """
    try:
        url = f"https://www.reddit.com/r/CryptoCurrency/search.json?q={query}&restrict_sr=1&sort=new&limit={limit}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) CryptoAnalyticsDashboard/1.0'}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code != 200:
            return 50.0 # Default neutral if Reddit blocks us
            
        data = response.json()
        posts = data.get('data', {}).get('children', [])
        
        if not posts:
            return 50.0
            
        analyzer = SentimentIntensityAnalyzer()
        total_compound = 0.0
        
        for post in posts:
            post_data = post['data']
            title = post_data.get('title', '')
            selftext = post_data.get('selftext', '')
            
            # Combine and clean
            text = clean_text(title + " " + selftext)
            
            if text.strip():
                scores = analyzer.polarity_scores(text)
                total_compound += scores['compound']
                
        avg_compound = total_compound / len(posts)
        
        # VADER compound is between -1 (Extreme Fear) and +1 (Extreme Greed)
        # Normalize to 0-100 scale: (score + 1) * 50
        fear_greed_score = (avg_compound + 1.0) * 50.0
        
        return max(0.0, min(100.0, fear_greed_score))
        
    except Exception as e:
        print(f"Error fetching Reddit sentiment: {e}")
        return 50.0
