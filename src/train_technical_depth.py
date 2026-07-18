# ============================================================
# Compare MiniLM, SciBERT, SPECTER2, BGE, and TF-IDF
# for predicting annotated Technical Depth
#
# Inputs:
#   1. Annotated sample:
#      data/raw/technical_depth_annotations.csv
#
#   2. Full 2,619-paper corpus:
#      data/processed/paper_features_2619.csv
#
# Required annotated columns:
#   - Paper ID
#   - Canonical Abstract
#   - Average Grade
#
# Output:
#   - model_comparison_metrics.csv
#   - all_2619_predicted_technical_depth_best_model.csv
#   - best_technical_depth_model.joblib
# ============================================================

# Install once if needed:
# !pip install -q sentence-transformers transformers scikit-learn scipy joblib pandas numpy torch

from pathlib import Path
import gc
import json
import random
import warnings

import joblib
import numpy as np
import pandas as pd
import torch

from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 4000

ANNOTATED_FILE = Path(
    PROJECT_ROOT / "data/raw/technical_depth_annotations.csv"
)

CORPUS_FILE = Path(
    PROJECT_ROOT / "data/processed/relevance_features_2619.csv"
)

ID_COL = "Paper ID"
TEXT_COL = "Canonical Abstract"
TARGET_COL = "Average Grade"

N_SPLITS = 5
BATCH_SIZE = 32

OUTPUT_METRICS = PROJECT_ROOT / "data/results/model_comparison_metrics.csv"
OUTPUT_PREDICTIONS = PROJECT_ROOT / "data/processed/paper_features_2619.csv"
OUTPUT_MODEL = PROJECT_ROOT / "models/best_technical_depth_model.joblib"
OUTPUT_REPORT = PROJECT_ROOT / "data/results/technical_depth_model_report.json"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


# ============================================================
# 2. MODEL DEFINITIONS
# ============================================================

EMBEDDING_MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",

    # Scientific BERT encoder
    "SciBERT": "allenai/scibert_scivocab_uncased",

    # Scientific document embedding model
    "SPECTER2": "allenai/specter2_base",

    # General-purpose BGE embedding model
    "BGE": "BAAI/bge-small-en-v1.5",
}

RIDGE_ALPHAS = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


# ============================================================
# 3. LOAD AND CLEAN DATA
# ============================================================

annotated = pd.read_csv(ANNOTATED_FILE, low_memory=False)
corpus = pd.read_csv(CORPUS_FILE, low_memory=False)

required_annotated = {ID_COL, TEXT_COL, TARGET_COL}
required_corpus = {ID_COL, TEXT_COL}

missing_annotated = required_annotated - set(annotated.columns)
missing_corpus = required_corpus - set(corpus.columns)

if missing_annotated:
    raise ValueError(
        f"Annotated file is missing columns: {missing_annotated}"
    )

if missing_corpus:
    raise ValueError(
        f"Corpus file is missing columns: {missing_corpus}"
    )

annotated = annotated.copy()
corpus = corpus.copy()

annotated[TEXT_COL] = (
    annotated[TEXT_COL]
    .fillna("")
    .astype(str)
    .str.strip()
)

corpus[TEXT_COL] = (
    corpus[TEXT_COL]
    .fillna("")
    .astype(str)
    .str.strip()
)

annotated[TARGET_COL] = pd.to_numeric(
    annotated[TARGET_COL],
    errors="coerce",
)

annotated = annotated[
    annotated[ID_COL].notna()
    & annotated[TEXT_COL].ne("")
    & annotated[TARGET_COL].notna()
].drop_duplicates(subset=ID_COL)

corpus = corpus[
    corpus[ID_COL].notna()
].drop_duplicates(subset=ID_COL)

missing_ids = set(annotated[ID_COL]) - set(corpus[ID_COL])

if missing_ids:
    raise ValueError(
        f"{len(missing_ids)} annotated Paper IDs are absent from the full corpus."
    )

print("Annotated rows:", len(annotated))
print("Corpus rows:", len(corpus))
print("Target range:", annotated[TARGET_COL].min(), annotated[TARGET_COL].max())


# ============================================================
# 4. METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):
    return {
        "RMSE": float(
            mean_squared_error(y_true, y_pred) ** 0.5
        ),
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "R2": float(
            r2_score(y_true, y_pred)
        ),
        "Pearson": float(
            pearsonr(y_true, y_pred).statistic
        ),
        "Spearman": float(
            spearmanr(y_true, y_pred).statistic
        ),
    }


# ============================================================
# 5. CROSS-VALIDATED RIDGE FOR DENSE EMBEDDINGS
# ============================================================

def evaluate_dense_embeddings(
    model_name,
    embeddings,
    targets,
    cv,
):
    results = []
    predictions_by_alpha = {}

    for alpha in RIDGE_ALPHAS:
        oof_predictions = np.zeros(len(targets), dtype=float)

        for train_idx, valid_idx in cv.split(embeddings):
            X_train = embeddings[train_idx]
            X_valid = embeddings[valid_idx]

            y_train = targets[train_idx]

            model = Pipeline([
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "ridge",
                    Ridge(alpha=alpha),
                ),
            ])

            model.fit(X_train, y_train)
            oof_predictions[valid_idx] = model.predict(X_valid)

        metrics = calculate_metrics(
            targets,
            oof_predictions,
        )

        results.append({
            "Model": model_name,
            "Alpha": alpha,
            **metrics,
        })

        predictions_by_alpha[alpha] = oof_predictions

    result_df = pd.DataFrame(results)

    best_row = result_df.sort_values(
        ["RMSE", "MAE"],
        ascending=[True, True],
    ).iloc[0]

    best_alpha = float(best_row["Alpha"])
    best_predictions = predictions_by_alpha[best_alpha]

    return result_df, best_alpha, best_predictions


# ============================================================
# 6. CROSS-VALIDATED TF-IDF
# ============================================================

def evaluate_tfidf(
    texts,
    targets,
    cv,
):
    results = []
    predictions_by_alpha = {}

    for alpha in RIDGE_ALPHAS:
        oof_predictions = np.zeros(len(targets), dtype=float)

        for train_idx, valid_idx in cv.split(texts):
            X_train = texts.iloc[train_idx]
            X_valid = texts.iloc[valid_idx]

            y_train = targets[train_idx]

            pipeline = Pipeline([
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.98,
                        max_features=30000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "ridge",
                    Ridge(alpha=alpha),
                ),
            ])

            pipeline.fit(X_train, y_train)
            oof_predictions[valid_idx] = pipeline.predict(X_valid)

        metrics = calculate_metrics(
            targets,
            oof_predictions,
        )

        results.append({
            "Model": "TF-IDF",
            "Alpha": alpha,
            **metrics,
        })

        predictions_by_alpha[alpha] = oof_predictions

    result_df = pd.DataFrame(results)

    best_row = result_df.sort_values(
        ["RMSE", "MAE"],
        ascending=[True, True],
    ).iloc[0]

    best_alpha = float(best_row["Alpha"])
    best_predictions = predictions_by_alpha[best_alpha]

    return result_df, best_alpha, best_predictions


# ============================================================
# 7. GENERATE EMBEDDINGS
# ============================================================

annotated_texts = annotated[TEXT_COL].tolist()
targets = annotated[TARGET_COL].to_numpy(dtype=float)

cv = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

all_metric_rows = []
best_configs = {}
annotated_embeddings = {}

for short_name, huggingface_name in EMBEDDING_MODELS.items():

    print("\n" + "=" * 70)
    print("Embedding model:", short_name)
    print("Hugging Face model:", huggingface_name)
    print("=" * 70)

    model = SentenceTransformer(
        huggingface_name,
        device=DEVICE,
    )

    embeddings = model.encode(
        annotated_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("Embedding shape:", embeddings.shape)

    annotated_embeddings[short_name] = embeddings

    result_df, best_alpha, best_oof = evaluate_dense_embeddings(
        model_name=short_name,
        embeddings=embeddings,
        targets=targets,
        cv=cv,
    )

    all_metric_rows.append(result_df)

    best_configs[short_name] = {
        "alpha": best_alpha,
        "oof_predictions": best_oof,
        "hf_model": huggingface_name,
        "embedding_dimension": int(embeddings.shape[1]),
    }

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 8. EVALUATE TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("Evaluating TF-IDF")
print("=" * 70)

tfidf_results, tfidf_best_alpha, tfidf_best_oof = evaluate_tfidf(
    texts=annotated[TEXT_COL],
    targets=targets,
    cv=cv,
)

all_metric_rows.append(tfidf_results)

best_configs["TF-IDF"] = {
    "alpha": tfidf_best_alpha,
    "oof_predictions": tfidf_best_oof,
}


# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

all_metrics = pd.concat(
    all_metric_rows,
    ignore_index=True,
)

all_metrics = all_metrics.sort_values(
    ["RMSE", "MAE", "Spearman"],
    ascending=[True, True, False],
).reset_index(drop=True)

all_metrics.to_csv(
    OUTPUT_METRICS,
    index=False,
)

print("\nComplete model comparison:")
print(all_metrics.to_string(index=False))

best_row = all_metrics.iloc[0]

best_model_name = best_row["Model"]
best_alpha = float(best_row["Alpha"])

print("\nBest model:", best_model_name)
print("Best alpha:", best_alpha)
print("Best RMSE:", best_row["RMSE"])
print("Best MAE:", best_row["MAE"])
print("Best R2:", best_row["R2"])
print("Best Pearson:", best_row["Pearson"])
print("Best Spearman:", best_row["Spearman"])


# ============================================================
# 10. TRAIN BEST MODEL ON ALL 300 LABELLED PAPERS
# ============================================================

corpus_texts = corpus[TEXT_COL].tolist()

if best_model_name == "TF-IDF":

    final_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                max_features=30000,
                sublinear_tf=True,
            ),
        ),
        (
            "ridge",
            Ridge(alpha=best_alpha),
        ),
    ])

    final_model.fit(
        annotated[TEXT_COL],
        targets,
    )

    corpus_predictions = final_model.predict(
        corpus[TEXT_COL]
    )

    saved_object = {
        "model_type": "TF-IDF",
        "pipeline": final_model,
        "target_column": TARGET_COL,
        "text_column": TEXT_COL,
        "id_column": ID_COL,
    }

else:

    hf_model_name = best_configs[best_model_name]["hf_model"]

    embedding_model = SentenceTransformer(
        hf_model_name,
        device=DEVICE,
    )

    train_embeddings = annotated_embeddings[best_model_name]

    corpus_embeddings = embedding_model.encode(
        corpus_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    final_regressor = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "ridge",
            Ridge(alpha=best_alpha),
        ),
    ])

    final_regressor.fit(
        train_embeddings,
        targets,
    )

    corpus_predictions = final_regressor.predict(
        corpus_embeddings
    )

    saved_object = {
        "model_type": best_model_name,
        "huggingface_model": hf_model_name,
        "regressor": final_regressor,
        "target_column": TARGET_COL,
        "text_column": TEXT_COL,
        "id_column": ID_COL,
        "normalize_embeddings": True,
    }


# ============================================================
# 11. CLIP AND NORMALIZE PREDICTIONS
# ============================================================

human_min = float(targets.min())
human_max = float(targets.max())

corpus_predictions = np.clip(
    corpus_predictions,
    human_min,
    human_max,
)

prediction_min = float(corpus_predictions.min())
prediction_max = float(corpus_predictions.max())

if prediction_max > prediction_min:
    technical_depth_norm = (
        corpus_predictions - prediction_min
    ) / (
        prediction_max - prediction_min
    )
else:
    technical_depth_norm = np.zeros(
        len(corpus_predictions)
    )


# ============================================================
# 12. ADD ANNOTATED AND OOF VALUES
# ============================================================

best_oof_predictions = best_configs[
    best_model_name
]["oof_predictions"]

annotation_map = annotated.set_index(
    ID_COL
)[TARGET_COL]

oof_map = pd.Series(
    best_oof_predictions,
    index=annotated[ID_COL],
)

output = corpus.copy()

output["Predicted Technical Depth"] = corpus_predictions
output["Technical Depth Norm"] = technical_depth_norm

output["Annotated Technical Depth"] = output[
    ID_COL
].map(annotation_map)

output["OOF Predicted Technical Depth"] = output[
    ID_COL
].map(oof_map)

output["Technical Depth Source"] = np.where(
    output["Annotated Technical Depth"].notna(),
    "annotated-training-sample",
    "model-predicted",
)

output["Technical Depth Model"] = best_model_name

output.to_csv(
    OUTPUT_PREDICTIONS,
    index=False,
)

joblib.dump(
    saved_object,
    OUTPUT_MODEL,
)


# ============================================================
# 13. SAVE REPORT
# ============================================================

report = {
    "training_rows": int(len(annotated)),
    "corpus_rows": int(len(corpus)),
    "target_column": TARGET_COL,
    "best_model": best_model_name,
    "best_alpha": best_alpha,
    "selection_rule": (
        "Lowest RMSE, then lowest MAE, then highest Spearman correlation"
    ),
    "best_metrics": {
        "RMSE": float(best_row["RMSE"]),
        "MAE": float(best_row["MAE"]),
        "R2": float(best_row["R2"]),
        "Pearson": float(best_row["Pearson"]),
        "Spearman": float(best_row["Spearman"]),
    },
    "human_score_range": [
        human_min,
        human_max,
    ],
    "predicted_score_range": [
        prediction_min,
        prediction_max,
    ],
    "outputs": {
        "metrics": str(OUTPUT_METRICS),
        "predictions": str(OUTPUT_PREDICTIONS),
        "model": str(OUTPUT_MODEL),
    },
}

OUTPUT_REPORT.write_text(
    json.dumps(
        report,
        indent=2,
    )
)

print("\nSaved:")
print(OUTPUT_METRICS)
print(OUTPUT_PREDICTIONS)
print(OUTPUT_MODEL)
print(OUTPUT_REPORT)