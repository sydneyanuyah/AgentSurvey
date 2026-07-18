# Relevance calculation with weighted RRF

This folder isolates the retrieval-relevance stage so it can be audited and reproduced independently from technical-depth prediction and CRITIC weighting.

## Input

`data/retrieval_ranks_2619.csv` contains one row per deduplicated paper and the column `Best Rank Per Query JSON`. The JSON maps each matched retrieval query to the paper's best rank for that query.

## Weighted reciprocal rank fusion

For paper $i$ and query $q$:

$$
RRF_{iq}=rac{w_q}{60+r_{iq}}
$$

where $r_{iq}$ is the paper's best rank and $w_q$ depends on the query family:

| Query family | Weight |
|---|---:|
| Explicit base query | 1.5 |
| Explicit taxonomy query | 1.2 |
| Implicit query | 1.0 |

The global retrieval score is:

$$
WeightedRRF_i=\sum_q rac{w_q}{60+r_{iq}}
$$

## Taxonomy relevance used by CRITIC

The six taxonomy scores are calculated from their corresponding explicit taxonomy query contributions:

- coordination
- communication
- planning and reasoning
- memory
- tool use
- evaluation and benchmarking

For example, if a paper appears at rank 39 for the coordination query:

$$
coordination_i=rac{1.2}{60+39}=0.012121\ldots
$$

The relevance criterion used in the final four-input CRITIC model is:

$$
max\_taxonomy\_score_i=\max_d(taxonomy\_score_{id})
$$

This preserves the paper's strongest taxonomy alignment. `Weighted RRF Score` is retained for retrieval auditing but is not directly passed into CRITIC.

## Files

```text
relevance/
├── config/rrf_config.json
├── data/retrieval_ranks_2619.csv
├── src/calculate_rrf_relevance.py
└── README.md
```

## Run independently

From the repository root:

```bash
python relevance/src/calculate_rrf_relevance.py
```

Generated files:

- `data/processed/relevance_features_2619.csv`
- `data/results/rrf_relevance_audit.json`

All query names, family assignments, weights, taxonomy mappings, and the RRF constant are declared in `config/rrf_config.json`, rather than embedded invisibly in the analysis notebook.
