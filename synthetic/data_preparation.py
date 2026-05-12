# synthetic/data_preparation.py
"""
Synthetic crisis discourse generation with social network structure,
event-driven timeline, and conversational threading.
Uses GPT-4o-mini for realistic tweet text generation.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from openai import OpenAI
from core.config import OPENAI_API_KEY
import random
from typing import List, Dict

client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================================
# SOCIAL NETWORK GENERATION
# ============================================================================

class SocialAccount:
    """Represents a social media account with followers and influence"""
    
    def __init__(self, account_id: str, account_type: str, follower_count: int, verified: bool = False):
        self.id = account_id
        self.type = account_type
        self.follower_count = follower_count
        self.verified = verified
        self.influence_score = self._calculate_influence()
    
    def _calculate_influence(self) -> float:
        base_influence = {
            'official': 0.9, 'journalist': 0.7, 'activist': 0.5, 'citizen': 0.3
        }
        verification_boost = 0.2 if self.verified else 0.0
        follower_boost = min(0.3, np.log10(self.follower_count + 1) / 10)
        return min(1.0, base_influence.get(self.type, 0.3) + verification_boost + follower_boost)


def generate_social_network(num_accounts: int = 50) -> List[SocialAccount]:
    """Generate a realistic social network with different account types"""
    accounts = []
    
    for i in range(5):
        accounts.append(SocialAccount(f'official_{i}', 'official', random.randint(10000, 100000), True))
    for i in range(10):
        accounts.append(SocialAccount(f'journalist_{i}', 'journalist', random.randint(5000, 50000), random.random() > 0.3))
    for i in range(15):
        accounts.append(SocialAccount(f'activist_{i}', 'activist', random.randint(1000, 20000), random.random() > 0.7))
    for i in range(num_accounts - 30):
        accounts.append(SocialAccount(f'citizen_{i}', 'citizen', random.randint(100, 5000), random.random() > 0.95))
    
    return accounts


# ============================================================================
# EVENT-DRIVEN DISCOURSE
# ============================================================================

class CrisisEvent:
    def __init__(self, hour: int, event_type: str, intensity: float, triggered_categories: List[str]):
        self.hour = hour
        self.type = event_type
        self.intensity = intensity
        self.triggered_categories = triggered_categories
        self.decay_rate = 0.5


def generate_crisis_timeline() -> List[CrisisEvent]:
    return [
        CrisisEvent(0, 'fire_escalation', 0.8, ['official_update', 'personal_story', 'community_help']),
        CrisisEvent(12, 'political_statement', 0.6, ['political_blame', 'misinformation']),
        CrisisEvent(24, 'community_mobilization', 0.7, ['community_help', 'personal_story']),
        CrisisEvent(36, 'containment_update', 0.5, ['official_update', 'personal_story']),
        CrisisEvent(48, 'fire_setback', 0.9, ['official_update', 'political_blame', 'personal_story', 'misinformation']),
        CrisisEvent(60, 'recovery_begins', 0.4, ['community_help', 'personal_story', 'official_update']),
    ]


# ============================================================================
# TWEET GENERATION WITH LLM
# ============================================================================

def generate_realistic_tweet(category: str, event_context: str = None) -> str:
    category_prompts = {
        'official_update': "Generate a realistic Twitter post from an official emergency services account during the Australian bushfire crisis. Include specific locations. Formal, informative, actionable. Under 280 chars. Tweet only.",
        'political_blame': "Generate a realistic angry Twitter post blaming politicians for the bushfire crisis. Frustrated but not offensive. Under 280 chars. Tweet only.",
        'personal_story': "Generate a realistic personal Twitter post from someone affected by the bushfire. Emotional, authentic. Under 280 chars. Tweet only.",
        'community_help': "Generate a realistic Twitter post offering help during the bushfire crisis. Specific locations/actions. Compassionate. Under 280 chars. Tweet only.",
        'misinformation': "Generate a realistic but CLEARLY FALSE conspiracy theory tweet about the bushfire. Plausible but identifiably false. Under 280 chars. Tweet only."
    }
    
    prompt = category_prompts.get(category, "Generate a tweet about Australian bushfires.")
    if event_context:
        prompt = f"{prompt}\n\nContext: {event_context}"
    
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=prompt,
            input="Generate the tweet now."
        )
        return response.output_text.strip()
    except Exception as e:
        print(f"Error generating tweet for {category}: {e}")
        return f"[Generated tweet for {category}]"


def generate_reply_tweet(parent_tweet: Dict, account: SocialAccount, agree: bool = True) -> str:
    parent_text = parent_tweet['text'][:100]
    if agree:
        prompt = f"Generate a short supportive reply (under 200 chars) to this bushfire tweet: '{parent_text}'."
    else:
        prompt = f"Generate a short critical reply (under 200 chars) to this bushfire tweet: '{parent_text}'."
    
    try:
        response = client.responses.create(model="gpt-4o-mini", instructions=prompt, input="Generate the reply now.")
        return response.output_text.strip()
    except:
        return f"@user {'Agreed!' if agree else 'I disagree with this.'}"


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(category: str, account: SocialAccount, event: CrisisEvent = None,
                    is_reply: bool = False, parent_engagement: int = 0) -> Dict:
    feature_profiles = {
        'official_update': {'anger': 0.1, 'authority': 0.9, 'urgency': 0.8, 'location_specific': 0.7},
        'political_blame': {'anger': 0.8, 'authority': 0.2, 'urgency': 0.4, 'location_specific': 0.3},
        'personal_story': {'anger': 0.3, 'authority': 0.1, 'urgency': 0.6, 'location_specific': 0.8},
        'community_help': {'anger': 0.1, 'authority': 0.2, 'urgency': 0.5, 'location_specific': 0.9},
        'misinformation': {'anger': 0.6, 'authority': 0.1, 'urgency': 0.7, 'location_specific': 0.4}
    }
    
    features = feature_profiles[category].copy()
    
    if account.verified:
        features['authority'] = min(1.0, features['authority'] + 0.3)
    elif account.type == 'citizen':
        features['authority'] = max(0.05, features['authority'] - 0.1)
    
    if event:
        features['urgency'] = min(1.0, features['urgency'] * event.intensity)
        if event.intensity > 0.7 and category in ['political_blame', 'misinformation']:
            features['anger'] = min(1.0, features['anger'] + 0.2)
    
    for key in ['anger', 'authority', 'urgency', 'location_specific']:
        features[key] += np.random.uniform(-0.15, 0.15)
        features[key] = max(0, min(1, features[key]))
    
    base_engagement = account.follower_count / 100
    verification_multiplier = 2.0 if account.verified else 1.0
    parent_boost = parent_engagement * 0.3 if is_reply else 0
    
    features['likes'] = int(np.random.exponential(base_engagement * verification_multiplier) + parent_boost)
    features['retweets'] = int(np.random.exponential(base_engagement * verification_multiplier * 0.4) + parent_boost * 0.5)
    features['replies'] = int(np.random.exponential(base_engagement * verification_multiplier * 0.2))
    
    return features


# ============================================================================
# CONVERSATIONAL STRUCTURE
# ============================================================================

def generate_reply_chain(parent_tweet: Dict, accounts: List[SocialAccount],
                        max_depth: int = 3, reply_probability: float = 0.3) -> List[Dict]:
    replies = []
    current_depth = 0
    
    while current_depth < max_depth and random.random() < (reply_probability * (0.5 ** current_depth)):
        weights = [acc.influence_score for acc in accounts]
        replying_account = random.choices(accounts, weights=weights, k=1)[0]
        agree = random.random() > 0.3
        
        reply_text = generate_reply_tweet(parent_tweet, replying_account, agree)
        
        if parent_tweet['category'] == 'official_update' and not agree:
            reply_category = 'political_blame'
        elif parent_tweet['category'] == 'misinformation' and agree:
            reply_category = 'misinformation'
        else:
            reply_category = parent_tweet['category']
        
        features = extract_features(reply_category, replying_account, is_reply=True,
                                   parent_engagement=parent_tweet['features']['likes'])
        
        reply = {
            'id': f"tweet_{parent_tweet['id']}_reply_{current_depth}",
            'text': reply_text,
            'category': reply_category,
            'author_type': replying_account.type,
            'author_id': replying_account.id,
            'verified': replying_account.verified,
            'timestamp': parent_tweet['timestamp'] + timedelta(minutes=random.randint(5, 120)),
            'features': features,
            'raw_features': {k: features[k] for k in ['anger', 'authority', 'urgency', 'location_specific']},
            'reply_to': parent_tweet['id'],
            'conversation_id': parent_tweet.get('conversation_id', parent_tweet['id'])
        }
        
        replies.append(reply)
        current_depth += 1
        
        if current_depth < max_depth:
            nested = generate_reply_chain(reply, accounts, max_depth - current_depth, reply_probability * 0.5)
            replies.extend(nested)
    
    return replies


def generate_retweet_cascade(seed_tweet: Dict, accounts: List[SocialAccount],
                             cascade_size: int = None) -> List[Dict]:
    retweets = []
    if cascade_size is None:
        base_cascade = seed_tweet['features']['retweets'] // 10
        cascade_size = random.randint(0, max(1, base_cascade))
    
    for i in range(cascade_size):
        retweeting_account = random.choice(accounts)
        retweet = {
            'id': f"tweet_{seed_tweet['id']}_rt_{i}",
            'text': f"RT @{seed_tweet.get('author_id', 'user')}: {seed_tweet['text']}",
            'category': seed_tweet['category'],
            'author_type': retweeting_account.type,
            'author_id': retweeting_account.id,
            'verified': retweeting_account.verified,
            'timestamp': seed_tweet['timestamp'] + timedelta(minutes=random.randint(10, 300)),
            'features': seed_tweet['features'].copy(),
            'raw_features': seed_tweet['raw_features'].copy(),
            'retweet_of': seed_tweet['id']
        }
        retweets.append(retweet)
    
    return retweets


# ============================================================================
# MAIN DATASET GENERATION
# ============================================================================

def create_synthetic_dataset(num_tweets=500, use_llm=True):
    """Create a synthetic bushfire dataset with social structure and event dynamics."""
    
    print("=" * 60)
    print("GENERATING SYNTHETIC CRISIS DISCOURSE")
    print("=" * 60)
    
    accounts = generate_social_network(num_accounts=50)
    print(f"\nSocial network: {len(accounts)} accounts")
    
    events = generate_crisis_timeline()
    print(f"Crisis timeline: {len(events)} events over 72 hours")
    
    all_tweets = []
    tweet_counter = 0
    start_time = datetime.now() - timedelta(hours=72)
    
    for event in events:
        event_time = start_time + timedelta(hours=event.hour)
        num_seeds = int(20 * event.intensity)
        
        print(f"\n  Event at Hour {event.hour}: {event.type} (intensity: {event.intensity:.2f}) -> {num_seeds} seeds")
        
        for i in range(num_seeds):
            category = random.choice(event.triggered_categories)
            weights = [acc.influence_score ** 2 for acc in accounts]
            author = random.choices(accounts, weights=weights, k=1)[0]
            
            event_context = f"{event.type} at hour {event.hour}"
            text = generate_realistic_tweet(category, event_context) if use_llm else f"[{category}] Tweet about {event.type}"
            features = extract_features(category, author, event)
            
            seed_tweet = {
                'id': f'tweet_{tweet_counter}',
                'text': text,
                'category': category,
                'author_type': author.type,
                'author_id': author.id,
                'verified': author.verified,
                'timestamp': event_time + timedelta(minutes=random.randint(0, 120)),
                'features': features,
                'raw_features': {k: features[k] for k in ['anger', 'authority', 'urgency', 'location_specific']},
                'conversation_id': f'tweet_{tweet_counter}',
                'event_trigger': event.type
            }
            
            all_tweets.append(seed_tweet)
            tweet_counter += 1
            
            if random.random() < 0.3:
                replies = generate_reply_chain(seed_tweet, accounts, max_depth=3)
                all_tweets.extend(replies)
                tweet_counter += len(replies)
            
            if random.random() < 0.2 and seed_tweet['features']['likes'] > 20:
                retweets = generate_retweet_cascade(seed_tweet, accounts)
                all_tweets.extend(retweets)
                tweet_counter += len(retweets)
    
    # Background noise
    num_background = max(0, num_tweets - len(all_tweets))
    for i in range(num_background):
        category = random.choice(['personal_story', 'community_help', 'official_update'])
        author = random.choice(accounts)
        text = generate_realistic_tweet(category) if use_llm else f"[{category}] Background tweet"
        features = extract_features(category, author)
        
        all_tweets.append({
            'id': f'tweet_{tweet_counter}',
            'text': text,
            'category': category,
            'author_type': author.type,
            'author_id': author.id,
            'verified': author.verified,
            'timestamp': start_time + timedelta(hours=random.randint(0, 72)),
            'features': features,
            'raw_features': {k: features[k] for k in ['anger', 'authority', 'urgency', 'location_specific']},
            'conversation_id': f'tweet_{tweet_counter}'
        })
        tweet_counter += 1
    
    df = pd.DataFrame(all_tweets)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nTotal tweets: {len(df)}")
    print(f"Categories:\n{df['category'].value_counts()}")
    print(f"Authors:\n{df['author_type'].value_counts()}")
    
    return df
