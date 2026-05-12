# Algorithmic Assemblages and Crisis Sensemaking

Replication code for the paper:

> **Algorithmic Assemblages and Crisis Sensemaking: A Computational Hermeneutic Approach**  
> *Under review, Organization Studies*

This repository contains the complete codebase for running assemblage simulations, computing field-level measures, and replicating the analysis pipeline on real or synthetic crisis discourse data.

---

## Overview

The study argues that different algorithmic platform configurations do not merely filter the same underlying discourse—they constitute ontologically distinct discursive fields from identical content. Three assemblage configurations are instantiated and applied to the same corpus of Australian bushfire tweets (2019–2021), and ten field-level measures are computed for each resulting field.

### Three Assemblages

| Assemblage | Moderation logic | Trending logic | Feed weight emphasis |
|---|---|---|---|
| **Viral-Optimized** | Light (misinformation only) | Engagement + anger boost | Anger (1.2×), virality (1.5×) |
| **Community-Safety** | Strict (removes political + misinformation) | Authority consensus | Authority (1.4×), urgency (1.0×) |
| **Public-Square** | Moderate | Geographic clustering | Location (1.3×), engagement (0.6×) |

### Ten Field-Level Measures (M1–M10)

| # | Measure | Code key | What it captures |
|---|---|---|---|
| M1 | Discourse fragmentation | `discourse_fragmentation.fragmentation_index` | Category entropy across the visible feed |
| M2 | Narrative coherence | `narrative_coherence.coherence_score` | Dominant-category clustering and emotional consistency |
| M3 | Voice inequality (Gini) | `voice_distribution.gini_coefficient` | Author-type concentration: 0 = equality, 1 = monopoly |
| M4 | Voice diversity entropy | `voice_distribution.voice_diversity` | Normalised Shannon entropy across author types |
| M5 | Authority concentration | `authority_concentration.concentration_score` | Share of high-authority (official/journalist) voices |
| M6 | Grassroots visibility ratio | `authority_concentration.grassroots_ratio` | Share of low-authority (citizen) voices |
| M7 | Epistemic diversity score | `epistemic_diversity.diversity_score` | Coverage of unique author-type × category profiles |
| M8 | Epistemic profile entropy | `epistemic_diversity.epistemic_entropy` | Normalised entropy across epistemic profiles |
| M9 | Systematic exclusion rate | `systematic_exclusions.exclusion_rate` | Proportion of corpus hidden; breakdown by type and category |
| M10 | Dominant emergent frame | `emergent_frames.dominant_frame` | Qualitative frame constituted by the visible feed |

---

## Repository Structure

```
core/
  assemblages.py       # Component classes + three assemblage configurations
  experiment.py        # Orchestration: runs all three assemblages, saves outputs
  field_analytics.py   # Computes all ten field-level measures
  config.py            # API key loading (reads from .env)
real_data/
  data_preparation.py  # Loads Twitter CSV, classifies accounts, extracts features
synthetic/
  data_preparation.py  # Generates synthetic crisis tweets via OpenAI API
run_real.py            # Entry point: real Twitter data
run_synthetic.py       # Entry point: synthetic data
```

---

## Installation

Requires Python 3.10+.

```bash
pip install pandas numpy scipy python-dotenv openai
```

For the real-data pipeline, copy `.env.example` to `.env` and add your OpenAI API key (used only by the synthetic pipeline):

```
OPENAI_API_KEY=your_key_here
```

---

## Usage

### Synthetic data (no corpus required)

Generates crisis discourse via GPT-4o-mini, then runs all three assemblages:

```bash
python run_synthetic.py                         # generate 1,000 tweets
python run_synthetic.py --num-tweets 500        # smaller run
python run_synthetic.py --no-llm                # placeholder text, no API call
```

### Real Twitter data

Requires the bushfire tracking corpus (see *Data Availability* below):

```bash
python run_real.py                                          # full corpus, originals only
python run_real.py --sample 5000                            # random sample of 5,000 tweets
python run_real.py --csv path/to/corpus.csv --output-dir results/real
```

Both pipelines write results to the output directory:

```
results/
  field_analytics.json        # all ten measures per assemblage
  experiment_summary.json     # cross-assemblage comparison
  cross_assemblage_scores.csv # per-tweet visibility across all three fields
```

---

## Data Availability

The real-data pipeline expects a CSV of tweets with columns including `full_text`, `user_screen_name`, `retweet_count`, `favorite_count`, `created_at`, and `is_retweet`.

The bushfire tracking corpus used in the paper (52,438 tweets, 2019–2021) is not included in this repository.

The synthetic pipeline requires no external data and can be used to replicate the experimental logic with LLM-generated crisis discourse.

---

## Adapting to a New Context

To apply the assemblage framework to a different crisis or platform context:

1. **Prepare your corpus** — supply a CSV with the columns listed above, or adapt `real_data/data_preparation.py` to your schema.
2. **Adjust assemblage configurations** — edit the component parameters in `core/assemblages.py` to reflect the moderation, trending, and curation logic of the platforms you are studying.
3. **Run** — `python run_real.py --csv your_corpus.csv`

The field-level measures in `core/field_analytics.py` are corpus-agnostic and require no modification.

---

## Citation

If you use this code, please cite:

```
[Citation to be added upon publication]
```

---

## License

MIT License. See `LICENSE` for details.
