# core/assemblages.py
"""
Algorithmic assemblage architecture: components and assemblage configurations.
Shared by both synthetic and real data pipelines.
"""
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os


# ============================================================================
# COMPONENT LAYER: Explicit algorithmic components as configurable rules
# ============================================================================

class ModerationComponent:
    """Explicitly filters and marks tweets for removal based on rules"""
    
    def __init__(self, name: str, rules: Dict[str, Any]):
        self.name = name
        self.rules = rules
        self.moderated_count = 0
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply moderation rules and return dataframe with moderation_status column"""
        df = df.copy()
        df['moderation_status'] = 'approved'
        
        for rule_name, rule_config in self.rules.items():
            if rule_config.get('enabled', True):
                condition = self._build_condition(rule_config)
                matches = condition(df)
                df.loc[matches, 'moderation_status'] = rule_config.get('action', 'flagged')
                self.moderated_count += matches.sum()
        
        return df
    
    def _build_condition(self, rule_config: Dict) -> callable:
        """Build a filtering condition from rule config"""
        if rule_config['type'] == 'threshold':
            feature = rule_config['feature']
            threshold = rule_config['threshold']
            operator = rule_config.get('operator', '>')
            
            if operator == '>':
                return lambda df, f=feature, t=threshold: df['raw_features'].apply(lambda x: x.get(f, 0)) > t
            elif operator == '<':
                return lambda df, f=feature, t=threshold: df['raw_features'].apply(lambda x: x.get(f, 0)) < t
            elif operator == '==':
                return lambda df, f=feature, t=threshold: df['raw_features'].apply(lambda x: x.get(f)) == t
        
        elif rule_config['type'] == 'category':
            categories = rule_config['categories']
            return lambda df, cats=categories: df['category'].isin(cats)
        
        return lambda df: pd.Series([False] * len(df))


class TrendingComponent:
    """Identifies and amplifies trending topics/patterns"""
    
    def __init__(self, name: str, rules: Dict[str, Any]):
        self.name = name
        self.rules = rules
        self.trending_topics = []
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect trending patterns and add trending_score"""
        df = df.copy()
        df['trending_boost'] = 0.0
        
        for trend_rule in self.rules.get('detection_rules', []):
            if trend_rule.get('enabled', True):
                boost = self._calculate_trend_boost(df, trend_rule)
                df['trending_boost'] += boost
        
        return df
    
    def _calculate_trend_boost(self, df: pd.DataFrame, rule: Dict) -> pd.Series:
        """Calculate trending boost based on rule"""
        rule_type = rule.get('type', 'engagement')
        
        if rule_type == 'engagement':
            engagement = (df['features'].apply(lambda x: x.get('likes', 0)) +
                         df['features'].apply(lambda x: x.get('retweets', 0)))
            threshold = rule.get('threshold', 30)
            boost_factor = rule.get('boost_factor', 0.5)
            return (engagement > threshold).astype(float) * boost_factor
        
        elif rule_type == 'category':
            categories = rule.get('categories', [])
            boost_factor = rule.get('boost_factor', 0.3)
            return (df['category'].isin(categories)).astype(float) * boost_factor
        
        elif rule_type == 'feature':
            feature = rule.get('feature')
            threshold = rule.get('threshold', 0.5)
            boost_factor = rule.get('boost_factor', 0.3)
            return (df['raw_features'].apply(lambda x: x.get(feature, 0)) > threshold).astype(float) * boost_factor
        
        return pd.Series([0.0] * len(df))


class FeedCuratorComponent:
    """Ranks and selects which tweets appear in the feed"""
    
    def __init__(self, name: str, ranking_rules: Dict[str, float]):
        self.name = name
        self.ranking_rules = ranking_rules
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score and rank tweets for feed visibility"""
        df = df.copy()
        df['feed_score'] = 0.0
        
        # Only score approved tweets
        approved = df[df['moderation_status'] == 'approved'].copy()
        
        for feature, weight in self.ranking_rules.items():
            if feature.endswith('_weight'):
                base_feature = feature.replace('_weight', '')
                
                if base_feature in ['engagement', 'virality']:
                    if base_feature == 'engagement':
                        score = (approved['features'].apply(lambda x: x.get('likes', 0)) +
                                approved['features'].apply(lambda x: x.get('replies', 0))) / 100
                    else:  # virality
                        score = approved['features'].apply(lambda x: x.get('retweets', 0)) / 50
                    df.loc[approved.index, 'feed_score'] += np.minimum(score, 1.0) * weight
                
                else:
                    score = approved['raw_features'].apply(lambda x: x.get(base_feature, 0))
                    df.loc[approved.index, 'feed_score'] += score * weight
        
        # Add trending boost
        df['feed_score'] += df.get('trending_boost', 0)
        
        # Add recency bonus
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            hours_old = (datetime.now() - timestamps).dt.total_seconds() / 3600
            recency_boost = np.maximum(0, 1 - (hours_old / 72)) * self.ranking_rules.get('recency_weight', 0)
            df['feed_score'] += recency_boost
        
        return df.sort_values('feed_score', ascending=False).reset_index(drop=True)


class SearchComponent:
    """Simulates search result ranking based on query intent"""
    
    def __init__(self, name: str, search_rules: Dict[str, Any]):
        self.name = name
        self.search_rules = search_rules
    
    def apply(self, df: pd.DataFrame, query_type: str = 'general') -> pd.DataFrame:
        """Rank results based on simulated search intent"""
        df = df.copy()
        
        rules = self.search_rules.get(query_type, {})
        df['search_score'] = 0.0
        
        for rule in rules.get('matching_rules', []):
            if df['category'].isin(rule.get('categories', [])).any():
                boost = rule.get('boost_factor', 1.0)
                df.loc[df['category'].isin(rule.get('categories', [])), 'search_score'] += boost
        
        return df.sort_values('search_score', ascending=False).reset_index(drop=True)


# ============================================================================
# ASSEMBLAGE LAYER: Base manager + three configurations
# ============================================================================

class AssemblageManager:
    """Base class that composes different algorithmic components"""
    
    def __init__(self, name: str, moderation: ModerationComponent, 
                 trending: TrendingComponent, curator: FeedCuratorComponent,
                 search: SearchComponent):
        self.name = name
        self.moderation = moderation
        self.trending = trending
        self.curator = curator
        self.search = search
        self.execution_log = []
    
    def process_tweets(self, tweets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main pipeline: Moderation → Trending Detection → Feed Curation
        """
        df = tweets_df.copy()
        
        # STAGE 1: MODERATION
        df = self.moderation.apply(df)
        stage1_approved = (df['moderation_status'] == 'approved').sum()
        stage1_removed = (df['moderation_status'] != 'approved').sum()
        
        self.execution_log.append({
            'stage': 'Moderation',
            'approved': stage1_approved,
            'removed': stage1_removed,
            'removal_categories': df[df['moderation_status'] != 'approved']['category'].value_counts().to_dict()
        })
        
        # STAGE 2: TRENDING DETECTION
        df = self.trending.apply(df)
        trending_boosted = (df['trending_boost'] > 0).sum()
        
        self.execution_log.append({
            'stage': 'Trending Detection',
            'boosted_count': trending_boosted,
            'avg_boost_strength': df['trending_boost'].mean()
        })
        
        # STAGE 3: FEED CURATION
        df = self.curator.apply(df)
        
        visibility_threshold = df['feed_score'].quantile(0.7)
        df['visible_in_feed'] = df['feed_score'] >= visibility_threshold
        visible_count = df['visible_in_feed'].sum()
        
        self.execution_log.append({
            'stage': 'Feed Curation',
            'visible_count': visible_count,
            'visibility_threshold': float(visibility_threshold),
            'visible_categories': df[df['visible_in_feed']]['category'].value_counts().to_dict(),
            'visible_avg_anger': df[df['visible_in_feed']]['raw_features'].apply(lambda x: x.get('anger', 0)).mean(),
            'visible_avg_authority': df[df['visible_in_feed']]['raw_features'].apply(lambda x: x.get('authority', 0)).mean(),
        })
        
        self._log_results(df)
        
        return df
    
    def _log_results(self, processed_df: pd.DataFrame):
        """Log the complete execution trace"""
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj
        
        results = {
            'assemblage': self.name,
            'timestamp': datetime.now().isoformat(),
            'execution_pipeline': self.execution_log,
            'moderation_rules': self.moderation.rules,
            'trending_rules': self.trending.rules,
            'curation_rules': self.curator.ranking_rules,
            'final_statistics': {
                'total_tweets': int(len(processed_df)),
                'visible_tweets': int(processed_df['visible_in_feed'].sum()),
                'removed_tweets': int((processed_df['moderation_status'] != 'approved').sum()),
                'avg_feed_score': float(processed_df['feed_score'].mean()),
            }
        }
        
        results = convert_types(results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('logs', exist_ok=True)
        with open(f'logs/{self.name}_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)


class ViralOptimizedAssemblage(AssemblageManager):
    """
    Assemblage designed for engagement/virality.
    - Minimal moderation (lets everything through)
    - Aggressive trending detection on engagement metrics
    - Feed ranking heavily favors anger and virality
    """
    
    def __init__(self):
        moderation = ModerationComponent(
            "Viral-Permissive-Moderation",
            rules={
                'explicit_ban': {
                    'enabled': True,
                    'type': 'category',
                    'categories': [],
                    'action': 'removed'
                }
            }
        )
        
        trending = TrendingComponent(
            "Viral-Aggressive-Trending",
            rules={
                'detection_rules': [
                    {'enabled': True, 'type': 'engagement', 'threshold': 20, 'boost_factor': 0.8},
                    {'enabled': True, 'type': 'feature', 'feature': 'anger', 'threshold': 0.6, 'boost_factor': 0.6},
                    {'enabled': True, 'type': 'category', 'categories': ['political_blame'], 'boost_factor': 0.5}
                ]
            }
        )
        
        curator = FeedCuratorComponent(
            "Viral-Engagement-Curator",
            ranking_rules={
                'engagement_weight': 1.2,
                'virality_weight': 1.5,
                'anger_weight': 0.8,
                'authority_weight': -0.3,
                'urgency_weight': 0.2,
                'location_specific_weight': -0.1,
                'recency_weight': 0.3
            }
        )
        
        search = SearchComponent(
            "Viral-Trending-Search",
            search_rules={
                'general': {
                    'matching_rules': [
                        {'categories': ['political_blame'], 'boost_factor': 2.0},
                        {'categories': ['personal_story'], 'boost_factor': 1.5},
                        {'categories': ['misinformation'], 'boost_factor': 1.2}
                    ]
                }
            }
        )
        
        super().__init__("Viral-Optimized", moderation, trending, curator, search)


class CommunitySafetyAssemblage(AssemblageManager):
    """
    Assemblage designed for safety and authoritative information.
    - Aggressive moderation (removes controversial content)
    - Trending detection favors official sources
    - Feed ranking heavily weights authority and accuracy
    """
    
    def __init__(self):
        moderation = ModerationComponent(
            "Safety-Aggressive-Moderation",
            rules={
                'remove_misinformation': {
                    'enabled': True, 'type': 'category',
                    'categories': ['misinformation'], 'action': 'removed'
                },
                'flag_political_blame': {
                    'enabled': True, 'type': 'category',
                    'categories': ['political_blame'], 'action': 'flagged'
                },
                'suppress_angry_content': {
                    'enabled': True, 'type': 'threshold',
                    'feature': 'anger', 'threshold': 0.7,
                    'operator': '>', 'action': 'flagged'
                }
            }
        )
        
        trending = TrendingComponent(
            "Safety-Authority-Trending",
            rules={
                'detection_rules': [
                    {'enabled': True, 'type': 'feature', 'feature': 'authority', 'threshold': 0.7, 'boost_factor': 0.9},
                    {'enabled': True, 'type': 'feature', 'feature': 'urgency', 'threshold': 0.6, 'boost_factor': 0.7}
                ]
            }
        )
        
        curator = FeedCuratorComponent(
            "Safety-Authority-Curator",
            ranking_rules={
                'authority_weight': 1.4,
                'urgency_weight': 1.0,
                'engagement_weight': -0.2,
                'virality_weight': -0.3,
                'anger_weight': -1.0,
                'location_specific_weight': 0.5,
                'recency_weight': 0.4
            }
        )
        
        search = SearchComponent(
            "Safety-Authority-Search",
            search_rules={
                'general': {
                    'matching_rules': [
                        {'categories': ['official_update'], 'boost_factor': 3.0},
                        {'categories': ['community_help'], 'boost_factor': 1.8},
                        {'categories': ['personal_story'], 'boost_factor': 0.5}
                    ]
                }
            }
        )
        
        super().__init__("Community-Safety", moderation, trending, curator, search)


class PublicSquareAssemblage(AssemblageManager):
    """
    Assemblage designed as a local public forum.
    - Moderate moderation (removes only extreme content)
    - Trending detection favors local relevance
    - Feed ranking balances diverse voices with community focus
    """
    
    def __init__(self):
        moderation = ModerationComponent(
            "PublicSquare-Permissive-Moderation",
            rules={
                'remove_illegal': {
                    'enabled': True, 'type': 'category',
                    'categories': [], 'action': 'removed'
                }
            }
        )
        
        trending = TrendingComponent(
            "PublicSquare-Location-Trending",
            rules={
                'detection_rules': [
                    {'enabled': True, 'type': 'feature', 'feature': 'location_specific', 'threshold': 0.7, 'boost_factor': 0.8},
                    {'enabled': True, 'type': 'category', 'categories': ['community_help'], 'boost_factor': 0.7},
                    {'enabled': True, 'type': 'engagement', 'threshold': 10, 'boost_factor': 0.4}
                ]
            }
        )
        
        curator = FeedCuratorComponent(
            "PublicSquare-Community-Curator",
            ranking_rules={
                'location_specific_weight': 1.3,
                'engagement_weight': 0.6,
                'authority_weight': 0.3,
                'urgency_weight': 0.8,
                'anger_weight': -0.2,
                'virality_weight': 0.1,
                'recency_weight': 0.5
            }
        )
        
        search = SearchComponent(
            "PublicSquare-Local-Search",
            search_rules={
                'general': {
                    'matching_rules': [
                        {'categories': ['community_help'], 'boost_factor': 2.5},
                        {'categories': ['personal_story'], 'boost_factor': 2.0},
                        {'categories': ['official_update'], 'boost_factor': 1.0}
                    ]
                }
            }
        )
        
        super().__init__("Public-Square", moderation, trending, curator, search)
