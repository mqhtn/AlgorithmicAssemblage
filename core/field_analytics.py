# core/field_analytics.py
"""
Field-level analytics for algorithmic assemblage research.
Measures emergent properties, interaction effects, and field dynamics.
Shared by both synthetic and real data pipelines.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from collections import Counter
import itertools
from scipy import stats


class FieldAnalytics:
    """Analyzes field-level organizing properties from assemblage outputs"""
    
    def __init__(self, assemblage_name: str):
        self.assemblage_name = assemblage_name
        self.metrics = {}
    
    def compute_all_metrics(self, visible_df: pd.DataFrame, full_df: pd.DataFrame) -> Dict:
        """Compute comprehensive field-level metrics"""
        self.metrics = {
            'assemblage': self.assemblage_name,
            'discourse_fragmentation': self.discourse_fragmentation(visible_df),
            'narrative_coherence': self.narrative_coherence(visible_df),
            'voice_distribution': self.voice_distribution(visible_df),
            'authority_concentration': self.authority_concentration(visible_df),
            'epistemic_diversity': self.epistemic_diversity(visible_df),
            'temporal_dynamics': self.temporal_dynamics(visible_df),
            'systematic_exclusions': self.systematic_exclusions(visible_df, full_df),
            'emergent_frames': self.emergent_frame_analysis(visible_df),
            'category_cooccurrence': self.category_cooccurrence(visible_df),
        }
        return self.metrics
    
    def discourse_fragmentation(self, df: pd.DataFrame) -> Dict:
        """Measure how fragmented the discourse is. High = people talking about different things."""
        if len(df) == 0:
            return {'fragmentation_index': 0.0, 'interpretation': 'no_data'}
        
        cat_dist = df['category'].value_counts(normalize=True)
        entropy = stats.entropy(cat_dist)
        max_entropy = np.log(len(cat_dist))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        anger_variance = df['raw_features'].apply(lambda x: x.get('anger', 0)).var()
        authority_variance = df['raw_features'].apply(lambda x: x.get('authority', 0)).var()
        
        fragmentation = (normalized_entropy + anger_variance + authority_variance) / 3
        interpretation = 'unified' if fragmentation < 0.3 else 'moderate' if fragmentation < 0.6 else 'fragmented'
        
        return {
            'fragmentation_index': float(fragmentation),
            'category_entropy': float(normalized_entropy),
            'anger_variance': float(anger_variance),
            'authority_variance': float(authority_variance),
            'interpretation': interpretation
        }
    
    def narrative_coherence(self, df: pd.DataFrame) -> Dict:
        """Do the visible tweets form a coherent narrative?"""
        if len(df) == 0:
            return {'coherence_score': 0.0, 'interpretation': 'no_data'}
        
        categories = df['category'].tolist()
        cluster_score = 0.0
        for i in range(len(categories) - 1):
            if categories[i] == categories[i+1]:
                cluster_score += 1
        clustering_coefficient = cluster_score / (len(categories) - 1) if len(categories) > 1 else 0
        
        anger_std = df['raw_features'].apply(lambda x: x.get('anger', 0)).std()
        authority_std = df['raw_features'].apply(lambda x: x.get('authority', 0)).std()
        consistency_score = 1 - np.mean([anger_std, authority_std])
        
        coherence = (clustering_coefficient + consistency_score) / 2
        interpretation = 'coherent' if coherence > 0.6 else 'mixed' if coherence > 0.4 else 'incoherent'
        
        return {
            'coherence_score': float(coherence),
            'clustering_coefficient': float(clustering_coefficient),
            'emotional_consistency': float(consistency_score),
            'interpretation': interpretation
        }
    
    def voice_distribution(self, df: pd.DataFrame) -> Dict:
        """Whose voices dominate? Gini coefficient: 0 = equality, 1 = one voice dominates."""
        if len(df) == 0:
            return {'gini_coefficient': 0.0, 'interpretation': 'no_data'}
        
        author_counts = df['author_type'].value_counts().values
        sorted_counts = np.sort(author_counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
        
        author_dist = df['author_type'].value_counts(normalize=True)
        voice_entropy = stats.entropy(author_dist)
        max_entropy = np.log(len(author_dist))
        voice_diversity = voice_entropy / max_entropy if max_entropy > 0 else 0
        
        interpretation = 'diverse' if gini < 0.4 else 'moderate' if gini < 0.7 else 'concentrated'
        
        return {
            'gini_coefficient': float(gini),
            'voice_diversity': float(voice_diversity),
            'dominant_voice': df['author_type'].mode()[0] if len(df) > 0 else 'none',
            'interpretation': interpretation
        }
    
    def authority_concentration(self, df: pd.DataFrame) -> Dict:
        """How much is discourse dominated by high-authority vs grassroots voices?"""
        if len(df) == 0:
            return {'concentration_score': 0.0, 'interpretation': 'no_data'}
        
        authority_scores = df['raw_features'].apply(lambda x: x.get('authority', 0))
        high_authority_count = (authority_scores > 0.7).sum()
        low_authority_count = (authority_scores < 0.3).sum()
        total = len(df)
        authority_ratio = high_authority_count / total if total > 0 else 0
        grassroots_ratio = low_authority_count / total if total > 0 else 0
        concentration = authority_ratio
        
        interpretation = 'authority_dominated' if concentration > 0.6 else 'balanced' if concentration > 0.3 else 'grassroots_dominated'
        
        return {
            'concentration_score': float(concentration),
            'authority_ratio': float(authority_ratio),
            'grassroots_ratio': float(grassroots_ratio),
            'avg_authority': float(authority_scores.mean()),
            'interpretation': interpretation
        }
    
    def epistemic_diversity(self, df: pd.DataFrame) -> Dict:
        """How many different types of knowledge/expertise are represented?"""
        if len(df) == 0:
            return {'diversity_score': 0.0, 'interpretation': 'no_data'}
        
        epistemic_profiles = df['author_type'] + '_' + df['category']
        unique_profiles = epistemic_profiles.nunique()
        possible_profiles = len(df['author_type'].unique()) * len(df['category'].unique())
        diversity_score = unique_profiles / possible_profiles if possible_profiles > 0 else 0
        
        profile_dist = epistemic_profiles.value_counts(normalize=True)
        entropy = stats.entropy(profile_dist)
        max_entropy = np.log(len(profile_dist)) if len(profile_dist) > 0 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        interpretation = 'highly_diverse' if diversity_score > 0.6 else 'moderate' if diversity_score > 0.3 else 'homogeneous'
        
        return {
            'diversity_score': float(diversity_score),
            'epistemic_entropy': float(normalized_entropy),
            'unique_profiles': int(unique_profiles),
            'interpretation': interpretation
        }
    
    def temporal_dynamics(self, df: pd.DataFrame) -> Dict:
        """How does the discourse evolve over time?"""
        if len(df) == 0 or 'timestamp' not in df.columns:
            return {'evolution_pattern': 'no_data', 'temporal_diversity': 0.0}
        
        df_sorted = df.sort_values('timestamp')
        n = len(df_sorted)
        early = df_sorted.iloc[:n//3]
        middle = df_sorted.iloc[n//3:2*n//3]
        late = df_sorted.iloc[2*n//3:]
        
        early_cats = set(early['category'].unique()) if len(early) > 0 else set()
        middle_cats = set(middle['category'].unique()) if len(middle) > 0 else set()
        late_cats = set(late['category'].unique()) if len(late) > 0 else set()
        temporal_diversity = len(early_cats.union(middle_cats).union(late_cats))
        
        early_anger = early['raw_features'].apply(lambda x: x.get('anger', 0)).mean() if len(early) > 0 else 0
        late_anger = late['raw_features'].apply(lambda x: x.get('anger', 0)).mean() if len(late) > 0 else 0
        anger_trend = 'increasing' if late_anger > early_anger + 0.1 else 'decreasing' if late_anger < early_anger - 0.1 else 'stable'
        
        early_auth = early['raw_features'].apply(lambda x: x.get('authority', 0)).mean() if len(early) > 0 else 0
        late_auth = late['raw_features'].apply(lambda x: x.get('authority', 0)).mean() if len(late) > 0 else 0
        authority_trend = 'increasing' if late_auth > early_auth + 0.1 else 'decreasing' if late_auth < early_auth - 0.1 else 'stable'
        
        return {
            'temporal_diversity': int(temporal_diversity),
            'anger_trajectory': anger_trend,
            'authority_trajectory': authority_trend,
            'early_anger': float(early_anger),
            'late_anger': float(late_anger),
            'early_authority': float(early_auth),
            'late_authority': float(late_auth)
        }
    
    def systematic_exclusions(self, visible_df: pd.DataFrame, full_df: pd.DataFrame) -> Dict:
        """What gets systematically hidden? Identifies unintended filtering effects."""
        if len(full_df) == 0:
            return {'exclusion_rate': 0.0, 'interpretation': 'no_data'}
        
        hidden_df = full_df[~full_df['id'].isin(visible_df['id'])]
        exclusion_rate = len(hidden_df) / len(full_df) if len(full_df) > 0 else 0
        
        full_dist = full_df['category'].value_counts(normalize=True)
        visible_dist = visible_df['category'].value_counts(normalize=True)
        
        exclusion_bias = {}
        for cat in full_dist.index:
            full_pct = full_dist.get(cat, 0)
            visible_pct = visible_dist.get(cat, 0)
            exclusion_bias[cat] = float(full_pct - visible_pct)
        
        most_excluded = max(exclusion_bias.items(), key=lambda x: x[1]) if exclusion_bias else ('none', 0)
        hidden_authors = hidden_df['author_type'].value_counts(normalize=True).to_dict() if len(hidden_df) > 0 else {}
        
        return {
            'exclusion_rate': float(exclusion_rate),
            'total_hidden': int(len(hidden_df)),
            'category_exclusion_bias': exclusion_bias,
            'most_excluded_category': most_excluded[0],
            'most_excluded_magnitude': float(most_excluded[1]),
            'hidden_author_distribution': hidden_authors
        }
    
    def emergent_frame_analysis(self, df: pd.DataFrame) -> Dict:
        """What narrative frames emerge from the visible content?"""
        if len(df) == 0:
            return {'dominant_frame': 'no_data', 'frame_strength': 0.0}
        
        avg_anger = df['raw_features'].apply(lambda x: x.get('anger', 0)).mean()
        avg_authority = df['raw_features'].apply(lambda x: x.get('authority', 0)).mean()
        avg_urgency = df['raw_features'].apply(lambda x: x.get('urgency', 0)).mean()
        dominant_category = df['category'].mode()[0] if len(df) > 0 else 'none'
        
        if avg_anger > 0.6 and dominant_category == 'political_blame':
            frame, strength = 'accountability_crisis', avg_anger
        elif avg_authority > 0.6 and dominant_category == 'official_update':
            frame, strength = 'institutional_management', avg_authority
        elif avg_urgency > 0.6 and dominant_category == 'community_help':
            frame, strength = 'community_resilience', avg_urgency
        elif dominant_category == 'personal_story':
            frame, strength = 'human_tragedy', 1 - avg_authority
        else:
            frame, strength = 'mixed_narrative', 0.5
        
        return {
            'dominant_frame': frame,
            'frame_strength': float(strength),
            'supporting_category': dominant_category,
            'emotional_tone': 'angry' if avg_anger > 0.5 else 'urgent' if avg_urgency > 0.5 else 'neutral'
        }
    
    def category_cooccurrence(self, df: pd.DataFrame) -> Dict:
        """Which discourse types appear together vs. separately?"""
        if len(df) < 2:
            return {'cooccurrence_network': {}, 'interpretation': 'insufficient_data'}
        
        categories = df.sort_values('timestamp')['category'].tolist() if 'timestamp' in df.columns else df['category'].tolist()
        
        window_size = 5
        cooccurrence = Counter()
        for i in range(len(categories) - window_size + 1):
            window = categories[i:i+window_size]
            pairs = list(itertools.combinations(set(window), 2))
            cooccurrence.update(pairs)
        
        total_pairs = sum(cooccurrence.values())
        normalized_cooccurrence = {f"{k[0]}__{k[1]}": v/total_pairs for k, v in cooccurrence.items()} if total_pairs > 0 else {}
        most_connected = max(normalized_cooccurrence.items(), key=lambda x: x[1]) if normalized_cooccurrence else (('none', 'none'), 0)
        
        return {
            'cooccurrence_network': normalized_cooccurrence,
            'most_connected_pair': most_connected[0],
            'connection_strength': float(most_connected[1]),
            'total_unique_connections': len(cooccurrence)
        }
    

def compare_assemblages(analytics_results: List[Dict]) -> Dict:
    """Compare field properties across assemblages to identify interaction effects."""
    if len(analytics_results) < 2:
        return {'comparison': 'insufficient_data'}
    
    comparison = {
        'assemblage_comparison': {},
        'key_differences': [],
        'emergent_insights': [],
        'interaction_effects': []
    }
    
    for metric in ['discourse_fragmentation', 'narrative_coherence', 'voice_distribution', 'authority_concentration', 'epistemic_diversity', 'systematic_exclusions']:
        values = {r['assemblage']: r.get(metric, {}) for r in analytics_results}
        comparison['assemblage_comparison'][metric] = values
    
    for result in analytics_results:
        assemblage = result['assemblage']
        exclusions = result.get('systematic_exclusions', {})
        most_excluded = exclusions.get('most_excluded_category', 'none')
        
        if 'Safety' in assemblage and most_excluded == 'community_help':
            comparison['emergent_insights'].append({
                'assemblage': assemblage,
                'insight': 'PARADOX: Safety assemblage excludes community help requests',
                'implication': 'Safety focus inadvertently reduces peer-to-peer coordination'
            })
        
        if 'Viral' in assemblage:
            auth_conc = result.get('authority_concentration', {}).get('interpretation', '')
            if auth_conc == 'grassroots_dominated':
                comparison['emergent_insights'].append({
                    'assemblage': assemblage,
                    'insight': 'Viral logic creates authority vacuum',
                    'implication': 'Official warnings may not break through during escalation'
                })
        
        frame = result.get('emergent_frames', {}).get('dominant_frame', '')
        voice_dist = result.get('voice_distribution', {}).get('interpretation', '')
        
        if frame == 'institutional_management' and voice_dist == 'concentrated':
            comparison['interaction_effects'].append({
                'assemblage': assemblage,
                'effect': 'Authority amplification + voice concentration -> one-way broadcast',
                'mechanism': 'Moderation + trending rules synergize to create top-down information flow'
            })
    
    return comparison
