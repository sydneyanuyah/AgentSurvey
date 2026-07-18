#!/usr/bin/env python3
"""Recalculate RRF retrieval relevance and six taxonomy relevance scores.

The input stores each paper's best rank for every retrieval query as JSON.
For a query q with family weight w_q and rank r_iq, its contribution is:

    contribution_iq = w_q / (k + r_iq)

The global Weighted RRF Score sums all query contributions. Each taxonomy
score uses only its corresponding explicit taxonomy query. The relevance
criterion passed to CRITIC is the maximum of the six taxonomy scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "relevance/data/retrieval_ranks_2619.csv"
DEFAULT_CONFIG = ROOT / "relevance/config/rrf_config.json"
DEFAULT_OUTPUT = ROOT / "data/processed/relevance_features_2619.csv"
DEFAULT_AUDIT = ROOT / "data/results/rrf_relevance_audit.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calculate weighted RRF and taxonomy relevance.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    return p.parse_args()


def parse_rank_json(value: object, row_number: int) -> dict[str, float]:
    if pd.isna(value) or str(value).strip() == "":
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid rank JSON at row {row_number}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Rank JSON at row {row_number} is not an object.")
    clean: dict[str, float] = {}
    for query, rank in parsed.items():
        rank = float(rank)
        if not np.isfinite(rank) or rank <= 0:
            raise ValueError(f"Invalid rank {rank!r} for {query!r} at row {row_number}.")
        clean[str(query)] = rank
    return clean


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    k = float(cfg["rrf_constant_k"])
    family_weights = cfg["query_family_weights"]
    queries = cfg["queries"]
    taxonomy_outputs = cfg["taxonomy_output_columns"]

    df = pd.read_csv(args.input, low_memory=False)
    required = {"Paper ID", "Best Rank Per Query JSON"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df["Paper ID"].isna().any() or df["Paper ID"].duplicated().any():
        raise ValueError("Paper ID must be complete and unique.")

    unknown_queries: set[str] = set()
    weighted_scores: list[float] = []
    taxonomy_values = {column: [] for column in taxonomy_outputs.values()}

    for row_number, raw in enumerate(df["Best Rank Per Query JSON"], start=2):
        ranks = parse_rank_json(raw, row_number)
        unknown_queries.update(set(ranks) - set(queries))

        total = 0.0
        dimension_scores = {dimension: 0.0 for dimension in taxonomy_outputs}
        for query, rank in ranks.items():
            if query not in queries:
                continue
            metadata = queries[query]
            weight = float(family_weights[metadata["family"]])
            contribution = weight / (k + rank)
            total += contribution
            dimension = metadata["dimension"]
            if dimension in dimension_scores:
                dimension_scores[dimension] += contribution

        weighted_scores.append(total)
        for dimension, column in taxonomy_outputs.items():
            taxonomy_values[column].append(dimension_scores[dimension])

    if unknown_queries:
        raise ValueError(
            "Queries found in the data but absent from rrf_config.json: "
            + ", ".join(sorted(unknown_queries))
        )

    df["Weighted RRF Score"] = weighted_scores
    for column, values in taxonomy_values.items():
        df[column] = values

    taxonomy_columns = list(taxonomy_outputs.values())
    df["max_taxonomy_score"] = df[taxonomy_columns].max(axis=1)
    df["Dominant Taxonomy"] = (
        df[taxonomy_columns].idxmax(axis=1).str.replace("_score", "", regex=False)
    )

    # Global RRF is retained for retrieval auditing. CRITIC uses only
    # max_taxonomy_score as its relevance criterion.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    audit = {
        "papers": int(len(df)),
        "rrf_constant_k": k,
        "query_family_weights": family_weights,
        "number_of_configured_queries": len(queries),
        "taxonomy_columns": taxonomy_columns,
        "critic_relevance_column": cfg["relevance_for_critic"],
        "weighted_rrf_range": [float(df["Weighted RRF Score"].min()), float(df["Weighted RRF Score"].max())],
        "max_taxonomy_score_range": [float(df["max_taxonomy_score"].min()), float(df["max_taxonomy_score"].max())],
        "output": str(args.output),
    }
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("RRF relevance calculation complete")
    print(f"Papers: {len(df):,}")
    print(f"Output: {args.output}")
    print(f"Audit:  {args.audit}")


if __name__ == "__main__":
    main()
