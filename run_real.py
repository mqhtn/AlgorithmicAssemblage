#!/usr/bin/env python3
# run_real.py
"""
Entry point: Run the assemblage experiment on REAL Twitter data.
Loads bushfire tracking data, classifies tweets, extracts features,
then processes through all three assemblages.

Usage:
    python run_real.py
    python run_real.py --csv bf_data/tracking_combined.csv --sample 5000
    python run_real.py --originals-only --output-dir results/real
"""
import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from real_data.data_preparation import load_and_prepare_real_data
from core.experiment import run_assemblage_experiment


def main():
    parser = argparse.ArgumentParser(description='Run assemblage experiment on real Twitter data')
    parser.add_argument('--csv', default='bf_data/tracking_combined.csv', help='Path to tracking CSV')
    parser.add_argument('--sample', type=int, default=None, help='Sample N tweets (None = use all)')
    parser.add_argument('--include-retweets', action='store_true', help='Include retweets (default: originals only)')
    parser.add_argument('--output-dir', default='results/real', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ALGORITHMIC ASSEMBLAGES — REAL TWITTER DATA EXPERIMENT")
    print("=" * 60)
    
    # Load and prepare real data
    df = load_and_prepare_real_data(
        csv_path=args.csv,
        filter_originals_only=not args.include_retweets,
        sample_size=args.sample
    )
    
    # Run experiment
    results = run_assemblage_experiment(df, output_dir=args.output_dir)
    
    # Show top tweets per assemblage
    print("\n" + "="*60)
    print("TOP TWEETS PER ASSEMBLAGE (REAL DATA)")
    print("="*60)
    
    for name, key in [('Viral-Optimized', 'viral_results'),
                       ('Community-Safety', 'safety_results'),
                       ('Public-Square', 'public_results')]:
        res = results[key]
        visible = res[res['visible_in_feed']].sort_values('feed_score', ascending=False)
        print(f"\n{name} — Top 5:")
        for _, row in visible.head(5).iterrows():
            print(f"  [{row['category']}] {str(row['text'])[:100]}... (score: {row['feed_score']:.2f})")


if __name__ == '__main__':
    main()
