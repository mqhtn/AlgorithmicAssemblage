#!/usr/bin/env python3
# run_synthetic.py
"""
Entry point: Run the assemblage experiment on SYNTHETIC data.
Generates crisis discourse with LLM, then processes through all three assemblages.

Usage:
    python run_synthetic.py
    python run_synthetic.py --num-tweets 500 --no-llm
"""
import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic.data_preparation import create_synthetic_dataset
from core.experiment import run_assemblage_experiment


def main():
    parser = argparse.ArgumentParser(description='Run assemblage experiment on synthetic data')
    parser.add_argument('--num-tweets', type=int, default=1000, help='Number of tweets to generate')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM generation (use placeholder text)')
    parser.add_argument('--output-dir', default='results/synthetic', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ALGORITHMIC ASSEMBLAGES — SYNTHETIC DATA EXPERIMENT")
    print("=" * 60)
    
    # Check for existing dataset
    csv_path = 'synthetic/bushfire_tweets_synthetic.csv'
    if os.path.exists(csv_path) and not args.no_llm:
        print(f"\nFound existing dataset at {csv_path}")
        import pandas as pd
        df = pd.read_csv(csv_path)
        df['raw_features'] = df['raw_features'].apply(eval)
        df['features'] = df['features'].apply(eval)
        print(f"Loaded {len(df)} tweets")
    else:
        print(f"\nGenerating {args.num_tweets} synthetic tweets...")
        df = create_synthetic_dataset(num_tweets=args.num_tweets, use_llm=not args.no_llm)
        os.makedirs('synthetic', exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")
    
    # Run experiment
    results = run_assemblage_experiment(df, output_dir=args.output_dir)
    
    # Show top tweets per assemblage
    print("\n" + "="*60)
    print("TOP TWEETS PER ASSEMBLAGE")
    print("="*60)
    
    for name, key in [('Viral-Optimized', 'viral_results'), 
                       ('Community-Safety', 'safety_results'),
                       ('Public-Square', 'public_results')]:
        res = results[key]
        visible = res[res['visible_in_feed']].sort_values('feed_score', ascending=False)
        print(f"\n{name} — Top 5:")
        for _, row in visible.head(5).iterrows():
            print(f"  [{row['category']}] {str(row['text'])[:80]}... (score: {row['feed_score']:.2f})")


if __name__ == '__main__':
    main()
