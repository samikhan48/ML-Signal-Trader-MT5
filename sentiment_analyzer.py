"""
FinBERT Sentiment Analyzer - Production Version
Fetches forex news, analyzes sentiment, generates symbol-specific scores
"""
import pandas as pd
import numpy as np
import re
import os
import logging
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
log_file = os.path.join(os.path.dirname(__file__), 'sentiment_analyzer_log.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Output path
MT5_FILES_PATH = os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal\Common\Files")

# RSS Feeds
FOREX_RSS_FEEDS = {
    'ForexLive': 'https://www.forexlive.com/feed/',
    'DailyFX': 'https://www.dailyfx.com/feeds/forex-market-news',
    'FXStreet': 'https://www.fxstreet.com/rss/news',
}

# =============================================================================
# KEYWORD DICTIONARIES
# =============================================================================

BULLISH_KEYWORDS = [
    'rally', 'rallies', 'surge', 'surges', 'soar', 'soars', 'jump', 'jumps',
    'gain', 'gains', 'rise', 'rises', 'climb', 'climbs', 'advance', 'advances',
    'strengthen', 'strengthens', 'higher', 'up', 'bullish', 'upside',
    'rate hike', 'hawkish', 'tightening', 'raises rates',
    'optimistic', 'positive', 'upgrade', 'upgrades', 'boost', 'boosts',
    'support', 'supports', 'buy', 'buying', 'demand',
]

BEARISH_KEYWORDS = [
    'fall', 'falls', 'drop', 'drops', 'plunge', 'plunges', 'tumble', 'tumbles',
    'decline', 'declines', 'sink', 'sinks', 'slide', 'slides', 'slump', 'slumps',
    'weaken', 'weakens', 'lower', 'down', 'bearish', 'downside',
    'rate cut', 'dovish', 'easing', 'cuts rates',
    'pessimistic', 'negative', 'downgrade', 'downgrades', 'concern', 'concerns',
    'fear', 'fears', 'sell', 'selling', 'pressure',
    'recession', 'crisis', 'crash', 'panic', 'risk-off',
]

# Skip patterns (irrelevant news)
SKIP_PATTERNS = [
    r'form\s*8k', r'form\s*8-k', r'form\s*13f', r'sec filing',
    r'ipo\s', r'merger', r'acquisition', r'stock split', r'dividend',
]

# Dynamic symbol detection patterns
FOREX_CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
MAJOR_STOCKS = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX']
COMMODITIES = ['GOLD', 'OIL', 'SILVER']
INDICES = ['SPX', 'SP500', 'NASDAQ', 'DOW']

# =============================================================================
# FINBERT MODEL LOADING
# =============================================================================

def load_finbert():
    """Load FinBERT model (lazy loading)"""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        logging.info("Loading FinBERT model...")
        MODEL_NAME = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logging.info(f"FinBERT loaded on {device}")
        
        return tokenizer, model, device
    except Exception as e:
        logging.error(f"Failed to load FinBERT: {e}")
        return None, None, None

# =============================================================================
# SENTIMENT ANALYSIS FUNCTIONS
# =============================================================================

def is_relevant(text):
    """Check if headline is relevant (not SEC filing, etc.)"""
    text_lower = text.lower()
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, text_lower):
            return False
    return True

def extract_symbols(text):
    """Extract mentioned symbols from headline"""
    symbols = set()
    text_upper = text.upper()
    
    # Forex currencies
    for curr in FOREX_CURRENCIES:
        if curr in text_upper:
            symbols.add(curr)
    
    # Major stocks
    for stock in MAJOR_STOCKS:
        if stock in text_upper:
            symbols.add(stock)
    
    # Commodities
    for comm in COMMODITIES:
        if comm in text_upper:
            symbols.add(comm)
    
    # Indices
    for idx in INDICES:
        if idx in text_upper:
            symbols.add(idx)
    
    return list(symbols)

def keyword_sentiment(text):
    """Rule-based sentiment using keywords"""
    text_lower = text.lower()
    
    bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
    total = bullish_count + bearish_count
    
    if total == 0:
        return 0, 0
    
    score = (bullish_count - bearish_count) / total
    confidence = min(total / 5, 1.0)
    
    return score, confidence

def finbert_sentiment(text, tokenizer, model, device):
    """Run FinBERT sentiment analysis"""
    if tokenizer is None or model is None:
        return 0, 0
    
    try:
        import torch
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        scores = probs[0].cpu().numpy()
        sentiment_score = float(scores[2] - scores[0])  # positive - negative
        
        return sentiment_score, float(max(scores))
    except Exception as e:
        logging.warning(f"FinBERT failed for headline: {e}")
        return 0, 0

def hybrid_sentiment(text, tokenizer, model, device):
    """Hybrid sentiment: keywords + FinBERT"""
    kw_score, kw_conf = keyword_sentiment(text)
    fb_score, fb_conf = finbert_sentiment(text, tokenizer, model, device)
    
    # Weight based on keyword confidence
    if kw_conf > 0.3:
        final_score = 0.7 * kw_score + 0.3 * fb_score
    elif kw_conf > 0:
        final_score = 0.5 * kw_score + 0.5 * fb_score
    else:
        final_score = fb_score
    
    return final_score

# =============================================================================
# NEWS FETCHING
# =============================================================================

def fetch_news(max_per_feed=20):
    """Fetch news from RSS feeds"""
    try:
        import feedparser
        from dateutil import parser as date_parser
    except ImportError:
        logging.error("feedparser or python-dateutil not installed!")
        return []
    
    all_news = []
    
    for source, url in FOREX_RSS_FEEDS.items():
        try:
            logging.info(f"Fetching from {source}...")
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:max_per_feed]:
                title = entry.get('title', '')
                published = entry.get('published', entry.get('updated', ''))
                
                try:
                    if published:
                        date = date_parser.parse(published)
                    else:
                        date = datetime.now()
                except:
                    date = datetime.now()
                
                if title:
                    all_news.append({
                        'title': title,
                        'date': date,
                        'source': source
                    })
        except Exception as e:
            logging.warning(f"{source} failed: {e}")
    
    return all_news

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_news():
    """Main sentiment analysis pipeline"""
    logging.info("=" * 60)
    logging.info("FINBERT SENTIMENT ANALYZER")
    logging.info("=" * 60)
    
    # Load FinBERT
    tokenizer, model, device = load_finbert()
    
    # Fetch news
    all_news = fetch_news(max_per_feed=20)
    logging.info(f"Fetched {len(all_news)} headlines")
    
    if not all_news:
        logging.warning("No news fetched!")
        return {}
    
    # Analyze each headline
    symbol_sentiments = defaultdict(list)
    
    for news in all_news:
        title = news['title']
        
        # Filter relevance
        if not is_relevant(title):
            continue
        
        # Extract symbols
        symbols = extract_symbols(title)
        if not symbols:
            continue
        
        # Get sentiment
        sentiment_score = hybrid_sentiment(title, tokenizer, model, device)
        
        # Assign to symbols
        for symbol in symbols:
            symbol_sentiments[symbol].append({
                'score': sentiment_score,
                'headline': title,
                'date': news['date']
            })
    
    # Aggregate by symbol
    results = []
    for symbol, sentiments in symbol_sentiments.items():
        scores = [s['score'] for s in sentiments]
        avg_score = np.mean(scores)
        confidence = min(len(scores) / 5, 1.0)  # More headlines = higher confidence
        
        results.append({
            'symbol': symbol,
            'sentiment_score': avg_score,
            'confidence': confidence,
            'headline_count': len(sentiments),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        logging.info(f"{symbol}: {avg_score:+.3f} (conf: {confidence:.2f}, headlines: {len(sentiments)})")
    
    return results

def save_sentiment_data(results):
    """Save sentiment data to CSV"""
    if not results:
        logging.warning("No sentiment data to save!")
        return
    
    df = pd.DataFrame(results)
    filepath = os.path.join(MT5_FILES_PATH, 'sentiment_data.csv')
    df.to_csv(filepath, index=False)
    logging.info(f"Saved {len(results)} symbols to: {filepath}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if not os.path.exists(MT5_FILES_PATH):
        os.makedirs(MT5_FILES_PATH, exist_ok=True)
    
    results = analyze_news()
    save_sentiment_data(results)
    
    logging.info("=" * 60)
    logging.info("DONE!")
    logging.info("=" * 60)
