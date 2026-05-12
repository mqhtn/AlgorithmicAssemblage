# core/experiment.py
"""
Shared experiment runner logic: processes tweets through assemblages,
runs field analytics, produces visualizations and results.
Used by both synthetic and real data pipelines.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

from core.assemblages import (
    ViralOptimizedAssemblage,
    CommunitySafetyAssemblage,
    PublicSquareAssemblage
)
from core.field_analytics import FieldAnalytics, compare_assemblages


def run_assemblage_experiment(df: pd.DataFrame, output_dir: str = 'results'):
    """
    Run all three assemblages on a prepared dataset and produce full analysis.
    
    Args:
        df: DataFrame with columns: id, text, category, author_type, timestamp,
            features (dict with likes/retweets/replies), raw_features (dict with anger/authority/urgency/location_specific)
        output_dir: Directory for results output
    
    Returns:
        dict with viral_results, safety_results, public_results, analytics
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print(f"Dataset: {len(df)} tweets")
    print(f"Category distribution:\n{df['category'].value_counts()}")
    
    # Initialize assemblages
    viral = ViralOptimizedAssemblage()
    safety = CommunitySafetyAssemblage()
    public = PublicSquareAssemblage()
    
    # Process through each assemblage
    print("\n" + "="*60)
    print("Processing through Viral-Optimized Assemblage...")
    viral_results = viral.process_tweets(df.copy())
    
    print("Processing through Community-Safety Assemblage...")
    safety_results = safety.process_tweets(df.copy())
    
    print("Processing through Public-Square Assemblage...")
    public_results = public.process_tweets(df.copy())
    
    # Run field-level analytics
    print("\n" + "="*60)
    print("RUNNING FIELD-LEVEL ANALYTICS")
    print("="*60)
    analytics = _run_field_analytics(df, viral_results, safety_results, public_results, output_dir)
    
    # Visualizations and comparison
    _analyze_results(df, viral_results, safety_results, public_results, analytics, output_dir)
    
    return {
        'viral_results': viral_results,
        'safety_results': safety_results,
        'public_results': public_results,
        'analytics': analytics
    }


def _run_field_analytics(original_df, viral_df, safety_df, public_df, output_dir):
    """Compute field-level metrics for each assemblage"""
    
    viral_visible = viral_df[viral_df['visible_in_feed']] if 'visible_in_feed' in viral_df.columns else viral_df
    safety_visible = safety_df[safety_df['visible_in_feed']] if 'visible_in_feed' in safety_df.columns else safety_df
    public_visible = public_df[public_df['visible_in_feed']] if 'visible_in_feed' in public_df.columns else public_df
    
    analytics_results = []
    
    for name, visible_df, full_df in [
        ('Viral-Optimized', viral_visible, viral_df),
        ('Community-Safety', safety_visible, safety_df),
        ('Public-Square', public_visible, public_df)
    ]:
        print(f"\n  Analyzing {name}...")
        analyzer = FieldAnalytics(name)
        metrics = analyzer.compute_all_metrics(visible_df, full_df)
        analytics_results.append(metrics)
        
        print(f"    Field Coherence Score: {metrics['field_coherence_score']:.3f}")
        print(f"    Discourse: {metrics['discourse_fragmentation']['interpretation']}")
        print(f"    Narrative: {metrics['narrative_coherence']['interpretation']}")
        print(f"    Voices: {metrics['voice_distribution']['interpretation']}")
        print(f"    Authority: {metrics['authority_concentration']['interpretation']}")
        print(f"    Frame: {metrics['emergent_frames']['dominant_frame']}")
    
    print("\n  Comparing assemblages for emergent effects...")
    comparison = compare_assemblages(analytics_results)
    
    output = {
        'individual_analytics': analytics_results,
        'comparative_analysis': comparison,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'field_analytics.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Field analytics saved to {output_dir}/field_analytics.json")
    return output


def _analyze_results(original_df, viral_df, safety_df, public_df, analytics_results, output_dir):
    """Visualizations and amplification/suppression analysis"""
    
    viral_visible = viral_df[viral_df['visible_in_feed']] if 'visible_in_feed' in viral_df.columns else viral_df
    safety_visible = safety_df[safety_df['visible_in_feed']] if 'visible_in_feed' in safety_df.columns else safety_df
    public_visible = public_df[public_df['visible_in_feed']] if 'visible_in_feed' in public_df.columns else public_df
    
    results = {
        'Original': original_df,
        'Viral-Optimized': viral_visible,
        'Community-Safety': safety_visible,
        'Public-Square': public_visible
    }
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (name, df) in enumerate(results.items()):
        if len(df) > 0:
            cat_dist = df['category'].value_counts(normalize=True)
            axes[idx].bar(cat_dist.index, cat_dist.values)
            axes[idx].set_title(f'{name}\n({len(df)} tweets visible)')
            axes[idx].set_ylabel('Proportion of Visible Content')
            axes[idx].tick_params(axis='x', rotation=45)
            
            if 'raw_features' in df.columns:
                avg_anger = df['raw_features'].apply(lambda x: x.get('anger', 0)).mean()
                avg_auth = df['raw_features'].apply(lambda x: x.get('authority', 0)).mean()
                axes[idx].text(0.05, 0.95, f"Avg Anger: {avg_anger:.2f}\nAvg Authority: {avg_auth:.2f}",
                             transform=axes[idx].transAxes, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'assemblage_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Amplification/suppression analysis
    print("\n" + "="*60)
    print("AMPLIFICATION/SUPPRESSION ANALYSIS")
    print("="*60)
    
    original_cats = original_df['category'].value_counts(normalize=True)
    
    for name, filtered_df in [('Viral', viral_visible), ('Safety', safety_visible), ('Public', public_visible)]:
        print(f"\n{name} Assemblage:")
        filtered_cats = filtered_df['category'].value_counts(normalize=True)
        
        for cat in original_cats.index:
            orig_pct = original_cats.get(cat, 0)
            filt_pct = filtered_cats.get(cat, 0)
            if orig_pct > 0:
                change = (filt_pct - orig_pct) / orig_pct
                symbol = "+" if change > 0 else ""
                print(f"  {cat}: {orig_pct:.1%} -> {filt_pct:.1%} ({symbol}{change:.0%})")
    
    # Save summary
    summary = {
        'experiment_date': datetime.now().isoformat(),
        'total_tweets': len(original_df),
        'viral_visible': len(viral_visible),
        'safety_visible': len(safety_visible),
        'public_visible': len(public_visible),
        'category_changes': {}
    }
    
    for cat in original_cats.index:
        summary['category_changes'][cat] = {
            'original': float(original_cats.get(cat, 0)),
            'viral': float(viral_visible['category'].value_counts(normalize=True).get(cat, 0)),
            'safety': float(safety_visible['category'].value_counts(normalize=True).get(cat, 0)),
            'public': float(public_visible['category'].value_counts(normalize=True).get(cat, 0))
        }
    
    with open(os.path.join(output_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to '{output_dir}/'")
    
    # Print emergent insights
    if analytics_results:
        comparison = analytics_results.get('comparative_analysis', {})
        
        insights = comparison.get('emergent_insights', [])
        if insights:
            print("\n" + "="*60)
            print("EMERGENT INSIGHTS & INTERACTION EFFECTS")
            print("="*60)
            for i, insight in enumerate(insights, 1):
                print(f"\n  {i}. {insight['assemblage']}:")
                print(f"     {insight['insight']}")
                print(f"     Implication: {insight['implication']}")
        
        effects = comparison.get('interaction_effects', [])
        if effects:
            print("\n  Component Interaction Effects:")
            for i, effect in enumerate(effects, 1):
                print(f"\n  {i}. {effect['assemblage']}:")
                print(f"     {effect['effect']}")
                print(f"     Mechanism: {effect['mechanism']}")
        
        coherence_comp = comparison.get('assemblage_comparison', {}).get('coherence', {})
        if coherence_comp:
            print(f"\n  Field Coherence: Most={coherence_comp.get('most_coherent', 'N/A')}, "
                  f"Least={coherence_comp.get('least_coherent', 'N/A')}, "
                  f"Range={coherence_comp.get('coherence_range', 0):.3f}")
