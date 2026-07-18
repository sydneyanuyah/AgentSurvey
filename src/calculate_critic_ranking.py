# ============================================================
# CRITIC weighting for the final 4-input paper score
#
# Inputs:
#   1. Taxonomy-based relevance
#   2. Predicted technical depth
#   3. Citation impact
#   4. Query coverage
#
# The CRITIC weights:
#   - are non-negative
#   - sum to 1
#
# Since all four criteria are normalized to [0,1],
# the weighted Final Score is also bounded within [0,1].
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 1. FILE SETTINGS
# ============================================================

INPUT_FILE = PROJECT_ROOT / "data/processed/paper_features_2619.csv"

OUTPUT_FILE = PROJECT_ROOT / "data/results/final_paper_rankings.csv"

WEIGHTS_FILE = PROJECT_ROOT / "data/results/critic_weights.csv"

CORRELATION_FILE = PROJECT_ROOT / "data/results/critic_correlation_matrix.csv"

REPORT_FILE = PROJECT_ROOT / "data/results/critic_report.json"


# ============================================================
# 2. COLUMN SETTINGS
# ============================================================

ID_COL = "Paper ID"

# Four and only four CRITIC inputs
RELEVANCE_COL = "max_taxonomy_score"
TECHNICAL_DEPTH_COL = "Technical Depth Norm"
CITATION_COL = "Max Citations"
QUERY_COUNT_COL = "Number of Queries Matched"


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
)

required_columns = {
    ID_COL,
    RELEVANCE_COL,
    TECHNICAL_DEPTH_COL,
    CITATION_COL,
    QUERY_COUNT_COL,
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        f"{sorted(missing_columns)}"
    )

df = df.copy()


# ============================================================
# 4. CONVERT INPUTS TO NUMERIC
# ============================================================

numeric_columns = [
    RELEVANCE_COL,
    TECHNICAL_DEPTH_COL,
    CITATION_COL,
    QUERY_COUNT_COL,
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# 5. CHECK PAPER IDS
# ============================================================

if df[ID_COL].isna().any():
    missing_id_count = int(
        df[ID_COL].isna().sum()
    )

    raise ValueError(
        f"{missing_id_count} rows have missing Paper IDs."
    )

if df[ID_COL].duplicated().any():
    duplicate_count = int(
        df[ID_COL].duplicated().sum()
    )

    raise ValueError(
        f"{duplicate_count} duplicated Paper IDs were found."
    )


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

# Missing citation counts are treated as zero citations.
df[CITATION_COL] = (
    df[CITATION_COL]
    .fillna(0)
    .clip(lower=0)
)

# Missing query counts are treated as zero matches.
df[QUERY_COUNT_COL] = (
    df[QUERY_COUNT_COL]
    .fillna(0)
    .clip(lower=0)
)

# Relevance and technical depth are substantive model criteria.
# Missing values are imputed using their medians.
if df[RELEVANCE_COL].notna().sum() == 0:
    raise ValueError(
        f"Column '{RELEVANCE_COL}' contains no valid numeric values."
    )

if df[TECHNICAL_DEPTH_COL].notna().sum() == 0:
    raise ValueError(
        f"Column '{TECHNICAL_DEPTH_COL}' contains no valid numeric values."
    )

df[RELEVANCE_COL] = df[
    RELEVANCE_COL
].fillna(
    df[RELEVANCE_COL].median()
)

df[TECHNICAL_DEPTH_COL] = df[
    TECHNICAL_DEPTH_COL
].fillna(
    df[TECHNICAL_DEPTH_COL].median()
)


# ============================================================
# 7. NORMALIZATION FUNCTION
# ============================================================

def min_max_normalize(series: pd.Series) -> pd.Series:
    """
    Normalize a numeric Series to [0,1].

    Formula:
        (x - min(x)) / (max(x) - min(x))

    If all values are identical, the criterion has no
    discriminatory power, so the function returns zeros.
    """

    minimum = float(series.min())
    maximum = float(series.max())

    if not np.isfinite(minimum):
        raise ValueError(
            f"Invalid minimum in column '{series.name}'."
        )

    if not np.isfinite(maximum):
        raise ValueError(
            f"Invalid maximum in column '{series.name}'."
        )

    if np.isclose(
        maximum,
        minimum,
    ):
        return pd.Series(
            np.zeros(
                len(series),
                dtype=float,
            ),
            index=series.index,
            name=series.name,
        )

    normalized = (
        series - minimum
    ) / (
        maximum - minimum
    )

    return normalized.clip(
        lower=0.0,
        upper=1.0,
    )


# ============================================================
# 8. CREATE THE FOUR NORMALIZED CRITERIA
# ============================================================

# ------------------------------------------------------------
# Criterion 1: Taxonomy-based relevance
# ------------------------------------------------------------

df["Relevance Norm"] = min_max_normalize(
    df[RELEVANCE_COL]
)


# ------------------------------------------------------------
# Criterion 2: Technical depth
# ------------------------------------------------------------

# Although this column is expected to already be normalized,
# it is normalized again defensively to guarantee [0,1].
df["Technical Depth Norm CRITIC"] = min_max_normalize(
    df[TECHNICAL_DEPTH_COL]
)


# ------------------------------------------------------------
# Criterion 3: Citation impact
# ------------------------------------------------------------

# Log transformation reduces domination by highly cited outliers.
df["Citation Log"] = np.log1p(
    df[CITATION_COL]
)

df["Citation Norm"] = min_max_normalize(
    df["Citation Log"]
)


# ------------------------------------------------------------
# Criterion 4: Query coverage
# ------------------------------------------------------------

df["Query Coverage Norm"] = min_max_normalize(
    df[QUERY_COUNT_COL]
)


# ============================================================
# 9. BUILD THE CRITIC DECISION MATRIX
# ============================================================

criteria_columns = [
    "Relevance Norm",
    "Technical Depth Norm CRITIC",
    "Citation Norm",
    "Query Coverage Norm",
]

X = df[
    criteria_columns
].astype(float).copy()


# ============================================================
# 10. VALIDATE NORMALIZED CRITERIA
# ============================================================

if X.isna().any().any():
    bad_columns = X.columns[
        X.isna().any()
    ].tolist()

    raise ValueError(
        "Missing values remain in normalized criteria: "
        f"{bad_columns}"
    )

if not np.isfinite(
    X.to_numpy()
).all():
    raise ValueError(
        "The CRITIC decision matrix contains infinite "
        "or invalid numeric values."
    )

outside_range = (
    (X < -1e-12)
    | (X > 1 + 1e-12)
)

if outside_range.any().any():
    bad_columns = X.columns[
        outside_range.any()
    ].tolist()

    raise ValueError(
        "The following normalized criteria contain values "
        f"outside [0,1]: {bad_columns}"
    )

# Protect against tiny floating-point errors.
X = X.clip(
    lower=0.0,
    upper=1.0,
)


# ============================================================
# 11. CALCULATE CRITIC STANDARD DEVIATIONS
# ============================================================

# Standard deviation represents the contrast or variability
# supplied by each criterion.
#
# ddof=0 treats the 2,619 papers as the complete decision set.
standard_deviation = X.std(
    axis=0,
    ddof=0,
)


# ============================================================
# 12. CALCULATE THE CORRELATION MATRIX
# ============================================================

correlation_matrix = X.corr(
    method="pearson"
)

# A constant criterion can produce undefined correlations.
# Such undefined off-diagonal correlations are set to zero,
# meaning the constant criterion is treated as uncorrelated.
correlation_matrix = correlation_matrix.fillna(
    0.0
)

# Each criterion is perfectly correlated with itself.
np.fill_diagonal(
    correlation_matrix.values,
    1.0,
)

correlation_matrix.to_csv(
    CORRELATION_FILE,
    index=True,
)


# ============================================================
# 13. CALCULATE CRITIC CONFLICT
# ============================================================

# For criterion j:
#
# conflict_j = sum_k (1 - r_jk)
#
# A criterion receives more conflict information when it is
# less redundant with the other criteria.
conflict = (
    1.0 - correlation_matrix
).sum(axis=1)


# ============================================================
# 14. CALCULATE INFORMATION CONTENT
# ============================================================

# CRITIC information content:
#
# C_j = sigma_j × sum_k(1 - r_jk)
#
# where:
#   sigma_j = standard deviation of criterion j
#   r_jk    = correlation between criteria j and k
information_content = (
    standard_deviation
    * conflict
)


# ============================================================
# 15. CALCULATE CRITIC WEIGHTS
# ============================================================

information_total = float(
    information_content.sum()
)

if not np.isfinite(
    information_total
):
    raise ValueError(
        "The total CRITIC information content is invalid."
    )

if information_total <= 0:
    raise ValueError(
        "CRITIC cannot calculate weights because the total "
        "information content is zero. Check whether one or "
        "more criteria are constant."
    )

critic_weights = (
    information_content
    / information_total
)

# Normalize once more to ensure numerical precision.
critic_weights = (
    critic_weights
    / critic_weights.sum()
)


# ============================================================
# 16. VERIFY THE WEIGHTS
# ============================================================

weight_sum = float(
    critic_weights.sum()
)

if not np.isclose(
    weight_sum,
    1.0,
    atol=1e-12,
):
    raise ValueError(
        "CRITIC weights do not sum to 1. "
        f"Current sum: {weight_sum}"
    )

if (
    critic_weights < -1e-12
).any():
    raise ValueError(
        "CRITIC generated at least one negative weight."
    )

critic_weights = critic_weights.clip(
    lower=0.0
)

critic_weights = (
    critic_weights
    / critic_weights.sum()
)

weight_sum = float(
    critic_weights.sum()
)


# ============================================================
# 17. CALCULATE WEIGHTED CONTRIBUTIONS
# ============================================================

df["Relevance Contribution"] = (
    X["Relevance Norm"]
    * critic_weights["Relevance Norm"]
)

df["Technical Depth Contribution"] = (
    X["Technical Depth Norm CRITIC"]
    * critic_weights["Technical Depth Norm CRITIC"]
)

df["Citation Contribution"] = (
    X["Citation Norm"]
    * critic_weights["Citation Norm"]
)

df["Query Coverage Contribution"] = (
    X["Query Coverage Norm"]
    * critic_weights["Query Coverage Norm"]
)


# ============================================================
# 18. CALCULATE FINAL SCORE
# ============================================================

df["Final Score"] = (
    df["Relevance Contribution"]
    + df["Technical Depth Contribution"]
    + df["Citation Contribution"]
    + df["Query Coverage Contribution"]
)

# Since every criterion is within [0,1] and all weights are
# non-negative and sum to 1, the Final Score must be in [0,1].
#
# Clipping only protects against tiny floating-point errors.
df["Final Score"] = df[
    "Final Score"
].clip(
    lower=0.0,
    upper=1.0,
)


# ============================================================
# 19. VERIFY FINAL SCORE
# ============================================================

final_score_min = float(
    df["Final Score"].min()
)

final_score_max = float(
    df["Final Score"].max()
)

if final_score_min < -1e-12:
    raise ValueError(
        "Final Score contains a value below zero."
    )

if final_score_max > 1 + 1e-12:
    raise ValueError(
        "Final Score contains a value above one."
    )


# ============================================================
# 20. GLOBAL RANK
# ============================================================

if "Global Rank" in df.columns:
    df = df.drop(
        columns=["Global Rank"]
    )

df["Global Rank"] = (
    df["Final Score"]
    .rank(
        method="min",
        ascending=False,
    )
    .astype(int)
)

df = df.sort_values(
    by=[
        "Final Score",
        ID_COL,
    ],
    ascending=[
        False,
        True,
    ],
).reset_index(
    drop=True
)


# ============================================================
# 21. OPTIONAL FIXED SCORE CLASSES
# ============================================================

# These are fixed score intervals.
# They are not equal-frequency quartiles.

df["Score Class"] = pd.cut(
    df["Final Score"],
    bins=[
        -np.inf,
        0.25,
        0.50,
        0.75,
        np.inf,
    ],
    labels=[
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    ],
    right=False,
)

df["Selected Class"] = np.where(
    df["Final Score"] >= 0.50,
    "Q3-Q4",
    "Q1-Q2",
)


# ============================================================
# 22. CREATE THE WEIGHT TABLE
# ============================================================

display_names = {
    "Relevance Norm": "Taxonomy Relevance",
    "Technical Depth Norm CRITIC": "Technical Depth",
    "Citation Norm": "Citation Impact",
    "Query Coverage Norm": "Query Coverage",
}

weight_table = pd.DataFrame({
    "Criterion Column": criteria_columns,

    "Criterion": [
        display_names[column]
        for column in criteria_columns
    ],

    "Standard Deviation": [
        float(
            standard_deviation[column]
        )
        for column in criteria_columns
    ],

    "Conflict": [
        float(
            conflict[column]
        )
        for column in criteria_columns
    ],

    "Information Content": [
        float(
            information_content[column]
        )
        for column in criteria_columns
    ],

    "CRITIC Weight": [
        float(
            critic_weights[column]
        )
        for column in criteria_columns
    ],
})

weight_table["Weight Sum"] = weight_sum

weight_table.to_csv(
    WEIGHTS_FILE,
    index=False,
)


# ============================================================
# 23. SAVE THE SCORED DATA
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# 24. CREATE REPORT
# ============================================================

class_counts = (
    df["Score Class"]
    .value_counts(
        dropna=False
    )
    .sort_index()
    .to_dict()
)

selected_class_counts = (
    df["Selected Class"]
    .value_counts(
        dropna=False
    )
    .to_dict()
)

report = {
    "number_of_papers": int(
        len(df)
    ),

    "input_columns": {
        "relevance": RELEVANCE_COL,
        "technical_depth": TECHNICAL_DEPTH_COL,
        "citations": CITATION_COL,
        "query_coverage": QUERY_COUNT_COL,
    },

    "normalized_criteria": {
        "relevance": "Relevance Norm",
        "technical_depth": "Technical Depth Norm CRITIC",
        "citations": "Citation Norm",
        "query_coverage": "Query Coverage Norm",
    },

    "normalization": {
        "relevance": "Min-max normalization",
        "technical_depth": "Min-max normalization",
        "citations": "log1p transformation followed by min-max normalization",
        "query_coverage": "Min-max normalization",
    },

    "weights": {
        display_names[criterion]: float(
            critic_weights[criterion]
        )
        for criterion in criteria_columns
    },

    "weight_sum": weight_sum,

    "final_score": {
        "minimum": float(
            df["Final Score"].min()
        ),
        "maximum": float(
            df["Final Score"].max()
        ),
        "mean": float(
            df["Final Score"].mean()
        ),
        "median": float(
            df["Final Score"].median()
        ),
        "standard_deviation": float(
            df["Final Score"].std(
                ddof=0
            )
        ),
    },

    "score_class_counts": {
        str(key): int(value)
        for key, value in class_counts.items()
    },

    "selected_class_counts": {
        str(key): int(value)
        for key, value in selected_class_counts.items()
    },

    "selected_class_rule": (
        "Q3-Q4 when Final Score is greater than or equal "
        "to 0.50; otherwise Q1-Q2."
    ),

    "outputs": {
        "scored_papers": str(
            OUTPUT_FILE
        ),
        "weights": str(
            WEIGHTS_FILE
        ),
        "correlation_matrix": str(
            CORRELATION_FILE
        ),
        "report": str(
            REPORT_FILE
        ),
    },
}

REPORT_FILE.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# 25. PRINT RESULTS
# ============================================================

print("\nCRITIC WEIGHTS")
print("=" * 70)

for criterion in criteria_columns:
    label = display_names[criterion]
    weight = critic_weights[criterion]

    print(
        f"{label:30s}: {weight:.12f}"
    )

print("-" * 70)

print(
    f"{'Total':30s}: {critic_weights.sum():.12f}"
)


print("\nFINAL SCORE SUMMARY")
print("=" * 70)

print(
    f"Minimum: {df['Final Score'].min():.12f}"
)

print(
    f"Maximum: {df['Final Score'].max():.12f}"
)

print(
    f"Mean:    {df['Final Score'].mean():.12f}"
)

print(
    f"Median:  {df['Final Score'].median():.12f}"
)


print("\nSCORE CLASS COUNTS")
print("=" * 70)

print(
    df["Score Class"]
    .value_counts()
    .sort_index()
)


print("\nCOMBINED CLASS COUNTS")
print("=" * 70)

print(
    df["Selected Class"]
    .value_counts()
)


print("\nTOP 10 PAPERS")
print("=" * 70)

top_columns = [
    ID_COL,
    RELEVANCE_COL,
    TECHNICAL_DEPTH_COL,
    CITATION_COL,
    QUERY_COUNT_COL,
    "Relevance Norm",
    "Technical Depth Norm CRITIC",
    "Citation Norm",
    "Query Coverage Norm",
    "Final Score",
    "Global Rank",
]

print(
    df[top_columns]
    .head(10)
    .to_string(
        index=False
    )
)


print("\nSAVED FILES")
print("=" * 70)

print(
    f"Scored papers:      {OUTPUT_FILE}"
)

print(
    f"CRITIC weights:     {WEIGHTS_FILE}"
)

print(
    f"Correlation matrix: {CORRELATION_FILE}"
)

print(
    f"JSON report:        {REPORT_FILE}"
)