# Agent Paper Ranking Pipeline

Reproducible pipeline for calculating retrieval relevance, predicting technical depth across 2,619 LLM-agent papers, and producing a final ranking with CRITIC objective weights.

## Repository structure

```text
agent-paper-ranking/
├── relevance/
│   ├── config/rrf_config.json
│   ├── data/retrieval_ranks_2619.csv
│   ├── src/calculate_rrf_relevance.py
│   └── README.md
├── config/project_config.json
├── data/
│   ├── raw/technical_depth_annotations.csv
│   ├── processed/
│   │   ├── relevance_features_2619.csv
│   │   └── paper_features_2619.csv
│   └── results/
├── models/
├── notebooks/
├── src/
│   ├── train_technical_depth.py
│   ├── calculate_critic_ranking.py
│   └── run_pipeline.py
├── requirements.txt
├── Makefile
└── README.md
```

## Pipeline

### 1. Retrieval relevance with weighted RRF

`relevance/src/calculate_rrf_relevance.py` reconstructs every query contribution from `Best Rank Per Query JSON`:

$$
RRF_{iq}=rac{w_q}{60+r_{iq}}
$$

Weights are 1.5 for the explicit base query, 1.2 for explicit taxonomy queries, and 1.0 for implicit queries. It calculates the global `Weighted RRF Score`, six taxonomy-specific relevance scores, and `max_taxonomy_score`. The final CRITIC model uses `max_taxonomy_score`; the global weighted RRF is retained as an auditable retrieval artifact. Full details are in [`relevance/README.md`](relevance/README.md).

### 2. Technical-depth prediction

`src/train_technical_depth.py` compares MiniLM, SciBERT, SPECTER2, BGE, and TF-IDF using shuffled five-fold cross-validation. Selection prioritizes lowest RMSE, then MAE, then highest Spearman correlation. The winning model is retrained on all 300 annotations and predicts the full corpus.

The original annotation file was renamed:

```text
dedup_ranked_outputs_sample_300_seed_4000 (1).csv
→ data/raw/technical_depth_annotations.csv
```

### 3. CRITIC ranking

`src/calculate_critic_ranking.py` uses exactly four normalized inputs:

1. `max_taxonomy_score`
2. `Technical Depth Norm`
3. `Max Citations`
4. `Number of Queries Matched`

CRITIC produces non-negative weights summing to 1; therefore, the weighted final score remains in `[0,1]`.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/run_pipeline.py
```

Individual stages:

```bash
python relevance/src/calculate_rrf_relevance.py
python src/train_technical_depth.py
python src/calculate_critic_ranking.py
```

Or:

```bash
make relevance
make train
make rank
make all
```

## Reproducibility controls

- Random seed: `4000`
- RRF constant: `60`
- Paper key: `Paper ID`
- Technical-depth target: `Average Grade`
- Query weights and mappings are versioned in `relevance/config/rrf_config.json`
- Scripts validate IDs, required columns, missing values, query configuration, normalized ranges, CRITIC weight sum, and final-score bounds
