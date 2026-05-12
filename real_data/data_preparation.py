# real_data/data_preparation.py
"""
Prepare real Twitter bushfire data for assemblage processing.
Loads tracking CSV data, classifies accounts, extracts features,
and produces a standardized DataFrame compatible with the assemblage pipeline.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import re
import json

# ============================================================================
# ACCOUNT TYPE CLASSIFICATION
# ============================================================================

def classify_account_type(row: pd.Series) -> str:
    """
    Classify a Twitter account into: official, journalist, activist, citizen
    based on UserDescription, UserVerified, UserScreenName, and follower count.
    """
    desc = str(row.get('UserDescription', '')).lower()
    screen_name = str(row.get('UserScreenName', '')).lower()
    verified = row.get('UserVerified', 0)
    followers = row.get('UserFollowersCount', 0)
    
    # Official/government accounts
    official_keywords = [
        'government', 'minister', 'council', 'official', 'department',
        'authority', 'bureau', 'police', 'ambulance', 'fire service',
        'emergency', 'rfs', 'cfa', 'ses', 'bom', 'premier', 'mp ',
        'member of parliament', 'senator', 'mayor'
    ]
    if any(kw in desc for kw in official_keywords):
        return 'official'
    if any(kw in screen_name for kw in ['gov', 'police', 'fire', 'emergency', 'bom_', 'rfs']):
        return 'official'
    
    # Journalist/media accounts
    journalist_keywords = [
        'journalist', 'reporter', 'editor', 'correspondent', 'anchor',
        'news', 'abc', 'sbs', 'media', 'press', 'broadcaster',
        'seven news', 'nine news', 'sky news', 'guardian', 'smh',
        'the australian', 'daily telegraph'
    ]
    if any(kw in desc for kw in journalist_keywords):
        return 'journalist'
    
    # Activist/organization accounts
    activist_keywords = [
        'activist', 'advocate', 'campaign', 'volunteer', 'ngo',
        'charity', 'foundation', 'climate', 'environment', 'conservation',
        'wildlife', 'rescue', 'community group', 'union', 'organis'
    ]
    if any(kw in desc for kw in activist_keywords):
        return 'activist'
    
    # Verified accounts with high followers that didn't match above -> journalist or official
    if verified and followers > 10000:
        return 'journalist'
    
    return 'citizen'


def classify_tweet_category(text: str) -> str:
    """
    Rule-based classification of tweet content into discourse categories.
    Uses keyword matching for speed and transparency (no black-box ML).
    """
    text_lower = text.lower()
    
    # Remove RT prefix for classification
    if text_lower.startswith('rt @'):
        # Find the colon after the username and classify the original text
        colon_idx = text_lower.find(': ')
        if colon_idx > 0:
            text_lower = text_lower[colon_idx + 2:]
    
    # Misinformation patterns
    misinfo_patterns = [
        r'arson', r'deliberately lit', r'set fire', r'conspiracy',
        r'cover.?up', r'fake news', r'hoax', r'false flag',
        r'5g', r'bill gates', r'agenda 21', r'cloud seeding',
        r'they don.t want you to know', r'wake up'
    ]
    if any(re.search(p, text_lower) for p in misinfo_patterns):
        return 'misinformation'
    
    # Political blame patterns
    political_patterns = [
        r'government fail', r'government.+blame', r'blame.+government',
        r'morrison', r'scomo', r'pm .+fail', r'prime minister',
        r'premier', r'politician', r'policy fail', r'funding cut',
        r'climate change.+government', r'government.+climate',
        r'incompeten', r'leadership fail', r'blood on .+ hands',
        r'political', r'labor|liberal|greens', r'parliament',
        r'minister', r'vote.+out'
    ]
    if any(re.search(p, text_lower) for p in political_patterns):
        return 'political_blame'
    
    # Official update patterns
    official_patterns = [
        r'warning|alert|advisory', r'fire ban', r'evacuat',
        r'containment', r'fire danger', r'catastrophic',
        r'emergency management', r'fire update', r'rfs|cfa|ses',
        r'road clos', r'shelter in place', r'watch and act',
        r'fire.+\d+%.+contained', r'hectares', r'fire front'
    ]
    if any(re.search(p, text_lower) for p in official_patterns):
        return 'official_update'
    
    # Community help patterns
    help_patterns = [
        r'donat', r'volunteer', r'fund.?rais', r'gofundme',
        r'offering', r'free .+(accommodation|room|bed|food|water)',
        r'help.+(needed|wanted|available)', r'relief',
        r'supplies', r'shelter', r'red cross', r'salvos',
        r'how.+to.+help', r'pitch in', r'lend a hand',
        r'support.+community', r'rebuil'
    ]
    if any(re.search(p, text_lower) for p in help_patterns):
        return 'community_help'
    
    # Personal story patterns
    personal_patterns = [
        r'my (home|house|town|family|friend)', r'lost everything',
        r'scared|terrified|heartbroken', r'i (saw|watched|heard)',
        r'we (had to|were forced)', r'our (community|town|home)',
        r'hero', r'brav', r'firefighter', r'firie',
        r'devastat', r'tragic', r'unbeliev', r'incredible',
        r'rip|rest in peace', r'memorial', r'tribute',
        r'koala|animal|wildlife'
    ]
    if any(re.search(p, text_lower) for p in personal_patterns):
        return 'personal_story'
    
    # Default: classify based on dominant theme keywords
    theme_scores = {
        'official_update': sum(1 for p in [r'fire', r'burn', r'smoke', r'weather', r'temperature', r'wind'] if re.search(p, text_lower)),
        'community_help': sum(1 for p in [r'help', r'support', r'community', r'together'] if re.search(p, text_lower)),
        'personal_story': sum(1 for p in [r'feel', r'think', r'hope', r'pray', r'heart', r'love', r'miss'] if re.search(p, text_lower)),
        'political_blame': sum(1 for p in [r'should', r'must', r'need to', r'demand', r'action'] if re.search(p, text_lower)),
    }
    best = max(theme_scores, key=theme_scores.get)
    if theme_scores[best] > 0:
        return best
    
    return 'personal_story'  # Default fallback


# ============================================================================
# FEATURE EXTRACTION FROM REAL TWEETS
# ============================================================================

def extract_features_from_tweet(row: pd.Series, author_type: str) -> Dict:
    """
    Extract anger, authority, urgency, location_specific features from a real tweet.
    Uses rule-based heuristics for transparency and reproducibility.
    """
    text = str(row.get('Text', '')).lower()
    verified = row.get('UserVerified', 0)
    followers = row.get('UserFollowersCount', 0)
    retweet_count = row.get('RetweetCount', 0)
    like_count = row.get('FavoriteCount / LikeCount', 0)
    
    # ANGER: detect angry/frustrated language
    anger_words = ['angry', 'furious', 'outraged', 'disgusting', 'shame',
                   'failure', 'incompetent', 'blame', 'pathetic', 'criminal',
                   'unacceptable', 'disgrace', 'utter', 'bloody', 'disaster',
                   'wtf', 'ffs', 'bullshit', 'unforgivable', 'negligent',
                   '!!!', '??!', 'scomo']
    anger_score = min(1.0, sum(1 for w in anger_words if w in text) * 0.2)
    # Exclamation marks boost anger
    anger_score = min(1.0, anger_score + text.count('!') * 0.05)
    # ALL CAPS words boost anger
    words = text.split()
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1)
    anger_score = min(1.0, anger_score + caps_ratio * 0.3)
    
    # AUTHORITY: based on account type and verification
    authority_map = {'official': 0.9, 'journalist': 0.7, 'activist': 0.4, 'citizen': 0.15}
    authority_score = authority_map.get(author_type, 0.2)
    if verified:
        authority_score = min(1.0, authority_score + 0.2)
    # High follower count adds authority
    if followers > 50000:
        authority_score = min(1.0, authority_score + 0.15)
    elif followers > 10000:
        authority_score = min(1.0, authority_score + 0.1)
    
    # URGENCY: detect urgent language
    urgency_words = ['urgent', 'emergency', 'immediately', 'now', 'critical',
                     'warning', 'danger', 'catastrophic', 'extreme', 'evacuate',
                     'life-threatening', 'act now', 'severe', 'alert']
    urgency_score = min(1.0, sum(1 for w in urgency_words if w in text) * 0.2)
    
    # LOCATION_SPECIFIC: detect location references
    location_patterns = [
        r'nsw|victoria|queensland|tasmania|sa |wa |act|nt ',
        r'sydney|melbourne|brisbane|canberra|adelaide|perth|hobart',
        r'bega|merimbula|eden|cobargo|mallacoota|kangaroo island',
        r'blue mountains|snowy|gippsland|south coast',
        r'my (town|area|region|community|suburb|street)',
    ]
    location_score = min(1.0, sum(0.25 for p in location_patterns if re.search(p, text)))
    
    # Add small noise for variance
    for name in ['anger', 'authority', 'urgency', 'location_specific']:
        pass  # We keep exact scores — this is real data, noise would be wrong
    
    features = {
        'anger': float(anger_score),
        'authority': float(authority_score),
        'urgency': float(urgency_score),
        'location_specific': float(location_score),
        'likes': int(like_count) if pd.notna(like_count) else 0,
        'retweets': int(retweet_count) if pd.notna(retweet_count) else 0,
        'replies': 0  # Not available in dataset
    }
    
    return features


# ============================================================================
# MAIN DATA PREPARATION
# ============================================================================

def load_and_prepare_real_data(csv_path: str = 'bf_data/tracking_combined.csv',
                                filter_originals_only: bool = True,
                                sample_size: int = None) -> pd.DataFrame:
    """
    Load real Twitter bushfire data and prepare it for the assemblage pipeline.
    
    Args:
        csv_path: Path to the tracking CSV file
        filter_originals_only: If True, remove retweets (keep only originals)
        sample_size: If set, randomly sample this many tweets
    
    Returns:
        DataFrame with standardized columns: id, text, category, author_type,
        timestamp, features, raw_features
    """
    print("=" * 60)
    print("LOADING REAL TWITTER BUSHFIRE DATA")
    print("=" * 60)
    
    # Load
    print(f"\nLoading {csv_path}...")
    raw_df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw tweets: {len(raw_df)}")
    
    # Filter to originals only (remove RTs)
    if filter_originals_only:
        is_rt = raw_df['Text'].astype(str).str.startswith('RT @')
        raw_df = raw_df[~is_rt].copy()
        print(f"  After removing retweets: {len(raw_df)}")
    
    # Sample if requested
    if sample_size and sample_size < len(raw_df):
        raw_df = raw_df.sample(n=sample_size, random_state=42).copy()
        print(f"  Sampled: {len(raw_df)}")
    
    # Classify account types
    print("\n  Classifying account types...")
    raw_df['author_type'] = raw_df.apply(classify_account_type, axis=1)
    print(f"  Account types:\n{raw_df['author_type'].value_counts().to_string()}")
    
    # Classify tweet categories
    print("\n  Classifying tweet categories...")
    raw_df['category'] = raw_df['Text'].astype(str).apply(classify_tweet_category)
    print(f"  Categories:\n{raw_df['category'].value_counts().to_string()}")
    
    # Extract features
    print("\n  Extracting features...")
    features_list = []
    for _, row in raw_df.iterrows():
        author_type = row['author_type']
        features = extract_features_from_tweet(row, author_type)
        features_list.append(features)
    
    # Build standardized DataFrame
    print("\n  Building standardized dataset...")
    prepared = pd.DataFrame({
        'id': raw_df['Id'].astype(str).values,
        'text': raw_df['Text'].values,
        'category': raw_df['category'].values,
        'author_type': raw_df['author_type'].values,
        'timestamp': pd.to_datetime(raw_df['CreatedAt']).values,
        'verified': raw_df['UserVerified'].astype(bool).values,
        'author_id': raw_df['UserScreenName'].values,
        'user_location': raw_df['UserLocation'].values,
        'follower_count': raw_df['UserFollowersCount'].values,
    })
    
    prepared['features'] = features_list
    prepared['raw_features'] = prepared['features'].apply(
        lambda x: {k: x[k] for k in ['anger', 'authority', 'urgency', 'location_specific']}
    )
    
    prepared = prepared.sort_values('timestamp').reset_index(drop=True)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("PREPARED DATASET SUMMARY")
    print("=" * 60)
    print(f"Total tweets: {len(prepared)}")
    print(f"\nCategory distribution:")
    print(prepared['category'].value_counts())
    print(f"\nAuthor type distribution:")
    print(prepared['author_type'].value_counts())
    print(f"\nVerified tweets: {prepared['verified'].sum()} ({prepared['verified'].mean()*100:.1f}%)")
    print(f"\nDate range: {prepared['timestamp'].min()} to {prepared['timestamp'].max()}")
    
    # Feature statistics
    print(f"\nFeature means:")
    for feat in ['anger', 'authority', 'urgency', 'location_specific']:
        vals = prepared['raw_features'].apply(lambda x: x.get(feat, 0))
        print(f"  {feat}: mean={vals.mean():.3f}, std={vals.std():.3f}")
    
    return prepared
