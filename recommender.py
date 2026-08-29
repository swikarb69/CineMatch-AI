"""
recommender.py
==============
CineMatch — four recommendation engines loaded lazily and cached for the
Streamlit runtime.

Engines
-------
- Popularity  : Bayesian weighted rating (IMDb formula)
- Content     : TF-IDF on genres + user tags, cosine similarity
- Collaborative: SVD matrix factorisation (scikit-surprise)
- Hybrid      : weighted blend of popularity + collaborative signals
"""

import html
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# ---------------------------------------------------------------------------
# Logging — replaces bare print() calls so Streamlit doesn't swallow them
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
DATA_DIR = _ROOT / "data"
MODEL_PATH = _ROOT / "models" / "svd_model.pkl"

# ---------------------------------------------------------------------------
# Hybrid scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------
POPULARITY_WEIGHT = 0.30
COLLAB_WEIGHT = 0.70

# ---------------------------------------------------------------------------
# CSV dtypes — cuts ratings memory usage by ~50%
# ---------------------------------------------------------------------------
_RATINGS_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
}
_MOVIES_DTYPES = {
    "movieId": "int32",
}


# ===========================================================================
# DATA LOADING (cached across Streamlit reruns)
# ===========================================================================

@st.cache_data(show_spinner=False)
def _load_data():
    """Load and minimally preprocess the three source CSVs."""
    log.info("Loading datasets…")

    movies_file = DATA_DIR / "movies.csv"
    ratings_file = DATA_DIR / "ratings.csv"
    tags_file = DATA_DIR / "tags.csv"

    if not movies_file.exists():
        raise FileNotFoundError(
            f"Movies dataset not found at '{movies_file}'. "
            "Please ensure the MovieLens dataset CSV files are placed in the 'data/' directory."
        )
    if not ratings_file.exists():
        raise FileNotFoundError(
            f"Ratings dataset not found at '{ratings_file}'. "
            "Please ensure 'ratings.csv' is placed in the 'data/' directory."
        )

    movies = pd.read_csv(
        movies_file,
        dtype=_MOVIES_DTYPES,
        encoding="utf-8",
    )

    # Drop timestamp immediately — saves memory on large ratings files
    ratings = pd.read_csv(
        ratings_file,
        dtype=_RATINGS_DTYPES,
        usecols=["userId", "movieId", "rating"],
        encoding="utf-8",
    )

    if tags_file.exists():
        tags = pd.read_csv(
            tags_file,
            usecols=["movieId", "tag"],
            encoding="utf-8",
            dtype={"movieId": "int32"},
        )
    else:
        tags = pd.DataFrame(columns=["movieId", "tag"], dtype=object)

    log.info(
        "Loaded — movies: %s  ratings: %s  tags: %s",
        movies.shape, ratings.shape, tags.shape,
    )
    return movies, ratings, tags


# ===========================================================================
# POPULARITY MODEL
# ===========================================================================

@st.cache_data(show_spinner=False)
def _build_popularity_model(movies: pd.DataFrame, ratings: pd.DataFrame):
    """
    IMDb Bayesian weighted rating:
        score = (v / (v + m)) * R  +  (m / (v + m)) * C
    where
        R = movie's average rating
        v = movie's vote count
        C = global mean rating across all movies
        m = 90th-percentile vote count (minimum-votes threshold)
    """
    log.info("Building popularity model…")

    movie_stats = (
        ratings
        .groupby("movieId", sort=False)
        .agg(avg_rating=("rating", "mean"), num_ratings=("rating", "count"))
        .reset_index()
    )

    C = float(movie_stats["avg_rating"].mean())
    m = float(movie_stats["num_ratings"].quantile(0.90))

    v = movie_stats["num_ratings"]
    movie_stats["score"] = (v / (v + m)) * movie_stats["avg_rating"] + (m / (v + m)) * C

    qualified = movie_stats[movie_stats["num_ratings"] >= m].copy()
    qualified = qualified.merge(movies[["movieId", "title"]], on="movieId")
    qualified = qualified.sort_values("score", ascending=False).reset_index(drop=True)

    log.info("Popularity model ready — %d qualifying movies.", len(qualified))
    return movie_stats, qualified


def get_top_movies(n: int = 10) -> pd.DataFrame:
    """Return the top-n movies by Bayesian weighted popularity score."""
    return _top_movies[["title", "avg_rating", "num_ratings", "score"]].head(n).reset_index(drop=True)


# ===========================================================================
# CONTENT-BASED MODEL
# ===========================================================================

@st.cache_resource(show_spinner=False)
def _build_content_model(movies: pd.DataFrame, ratings: pd.DataFrame, tags: pd.DataFrame):
    """
    Build a TF-IDF matrix over the combined genres + user-tag corpus.
    `indices` maps movie title → integer row position in content_movies.
    """
    log.info("Building content-based model…")

    if not tags.empty and "tag" in tags.columns:
        tags_clean = tags.dropna(subset=["tag"]).copy()
        tags_clean["tag"] = (
            tags_clean["tag"]
            .astype(str)
            .str.encode("ascii", errors="ignore")
            .str.decode("ascii")
            .str.strip()
        )

        movie_tags = (
            tags_clean
            .groupby("movieId", sort=False)["tag"]
            .agg(" ".join)
            .reset_index()
        )
    else:
        movie_tags = pd.DataFrame(columns=["movieId", "tag"])

    rating_stats = (
        ratings
        .groupby("movieId", sort=False)
        .agg(avg_rating=("rating", "mean"), num_ratings=("rating", "count"))
        .reset_index()
    )

    content_movies = (
        movies
        .merge(movie_tags, on="movieId", how="left")
        .merge(rating_stats, on="movieId", how="left")
    )
    content_movies["tag"] = content_movies["tag"].fillna("")
    content_movies["genres"] = content_movies["genres"].fillna("")
    content_movies["avg_rating"] = content_movies["avg_rating"].fillna(0.0)
    content_movies["num_ratings"] = content_movies["num_ratings"].fillna(0).astype("int32")
    content_movies["content"] = content_movies["genres"] + " " + content_movies["tag"]

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=25000,
        dtype=np.float32,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(content_movies["content"])

    # Map title → positional index; keep first occurrence when duplicates exist
    indices = pd.Series(content_movies.index, index=content_movies["title"])
    indices = indices[~indices.index.duplicated(keep="first")]

    log.info("Content model ready — TF-IDF matrix: %s.", tfidf_matrix.shape)
    return content_movies, tfidf_matrix, indices


def search_movie(query: str) -> pd.DataFrame:
    """Return up to 20 movies whose title contains `query` (case-insensitive)."""
    if not query or not query.strip():
        return pd.DataFrame(columns=["title"])
    mask = _content_movies["title"].str.contains(query.strip(), case=False, na=False, regex=False)
    return _content_movies.loc[mask, ["title"]].head(20)


def recommend_content(
    movie_title: str,
    n: int = 10,
    min_ratings: int = 50,
) -> pd.DataFrame:
    """
    Return n movies similar to `movie_title` by TF-IDF cosine similarity,
    filtered to movies with at least `min_ratings` user ratings.
    """
    if movie_title not in _indices:
        return pd.DataFrame({"Error": [f"'{html.escape(movie_title)}' not found in the catalogue."]})

    idx = _indices[movie_title]

    sim_scores = linear_kernel(_tfidf_matrix[idx], _tfidf_matrix).flatten()

    # Fast top-k candidate retrieval using np.argpartition
    pool_size = min(n * 10 + 1, len(sim_scores))
    top_indices = np.argpartition(sim_scores, -pool_size)[-pool_size:]
    top_indices = top_indices[np.argsort(-sim_scores[top_indices])]
    top_indices = top_indices[top_indices != idx]  # exclude the query film itself

    candidates = _content_movies.iloc[top_indices][
        ["title", "genres", "avg_rating", "num_ratings"]
    ].copy()
    candidates["similarity"] = sim_scores[top_indices]

    candidates = candidates[candidates["num_ratings"] >= min_ratings]

    return (
        candidates
        .sort_values(["similarity", "avg_rating"], ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


# ===========================================================================
# COLLABORATIVE FILTERING (SVD)
# ===========================================================================

@st.cache_resource(show_spinner=False)
def _load_collab_model(movies: pd.DataFrame):
    """Load the pre-trained SVD model from disk and pre-compute item ID mappings."""
    log.info("Loading trained SVD model from %s…", MODEL_PATH)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"SVD model not found at '{MODEL_PATH}'. "
            "Run notebook 04_collaborative_filtering.ipynb to train and save it."
        )
    model = joblib.load(MODEL_PATH)
    movie_title_map = movies.set_index("movieId")["title"].to_dict()

    # Pre-compute raw item IDs array for fast vectorized lookup
    raw_iids = np.array(
        [int(model.trainset.to_raw_iid(i)) for i in range(model.trainset.n_items)],
        dtype="int32",
    )

    log.info("SVD model loaded successfully (%d items in trainset).", model.trainset.n_items)
    return model, movie_title_map, raw_iids


def recommend_for_user(user_id: int, n: int = 10) -> pd.DataFrame:
    """
    Return n movie recommendations for a known user using vectorized SVD scoring.

    Instead of calling model.predict() in a Python loop (slow — O(N) calls),
    we compute scores for *all* items in one matrix operation:
        score_i = clip(mu + bu + bi_i + qi_i · pu, 0.5, 5.0)
    This is 100–1000× faster on large item sets.
    """
    rated_movie_ids = set(
        _ratings.loc[_ratings["userId"] == user_id, "movieId"]
    )

    try:
        inner_uid = _model.trainset.to_inner_uid(user_id)
        mu = _model.trainset.global_mean
        bu = _model.bu[inner_uid]
        pu = _model.pu[inner_uid]

        # Vectorized: score every item at once
        all_scores = np.clip(
            mu + bu + _model.bi + _model.qi @ pu,
            0.5, 5.0,
        )
        score_series = pd.Series(all_scores, index=_raw_iids)

    except (ValueError, KeyError):
        # Unknown user — fall back to global mean for all items
        mu = _model.trainset.global_mean
        score_series = pd.Series(np.full(len(_raw_iids), mu), index=_raw_iids)

    # Filter out already-rated movies and keep top-n
    if rated_movie_ids:
        score_series = score_series[~score_series.index.isin(rated_movie_ids)]

    top = score_series.nlargest(n)

    return pd.DataFrame({
        "title": [_movie_title_map.get(mid, "Unknown") for mid in top.index],
        "predicted_rating": top.values.round(2),
    })


# ===========================================================================
# HYBRID MODEL
# ===========================================================================

@st.cache_data(show_spinner=False)
def _build_hybrid_frame(movies: pd.DataFrame, movie_stats: pd.DataFrame) -> pd.DataFrame:
    """Merge movie metadata with pre-computed popularity stats and normalise."""
    log.info("Building hybrid model frame…")

    hybrid = movies.merge(
        movie_stats[["movieId", "avg_rating", "num_ratings", "score"]],
        on="movieId",
        how="left",
    )
    hybrid["score"] = hybrid["score"].fillna(0.0)
    hybrid["avg_rating"] = hybrid["avg_rating"].fillna(0.0)
    hybrid["num_ratings"] = hybrid["num_ratings"].fillna(0).astype("int32")

    max_score = hybrid["score"].max()
    hybrid["popularity_norm"] = hybrid["score"] / max_score if max_score > 0 else 0.0

    log.info("Hybrid frame ready — %d movies.", len(hybrid))
    return hybrid


def hybrid_recommend(
    user_id: int,
    n: int = 10,
    popularity_weight: float = POPULARITY_WEIGHT,
    collab_weight: float = COLLAB_WEIGHT,
) -> pd.DataFrame:
    """
    Hybrid recommendation: blend normalised popularity and collaborative scores.

    Parameters
    ----------
    user_id          : MovieLens user ID
    n                : number of results to return
    popularity_weight: weight for the popularity signal (default 0.30)
    collab_weight    : weight for the collaborative signal (default 0.70)
    """
    rated_movies = set(
        _ratings.loc[_ratings["userId"] == user_id, "movieId"]
    )

    candidates = _hybrid_movies[~_hybrid_movies["movieId"].isin(rated_movies)].copy()

    # --- Collaborative scores (vectorised) ----------------------------------
    try:
        inner_uid = _model.trainset.to_inner_uid(user_id)
        mu = _model.trainset.global_mean
        bu = _model.bu[inner_uid]
        pu = _model.pu[inner_uid]

        all_scores = np.clip(
            mu + bu + _model.bi + _model.qi @ pu,
            0.5, 5.0,
        )
        score_series = pd.Series(all_scores, index=_raw_iids)

        fallback = float(np.clip(mu + bu, 0.5, 5.0))
        candidates["collab_score"] = candidates["movieId"].map(score_series).fillna(fallback)

    except (ValueError, KeyError):
        # Cold-start: unknown user — use global mean so hybrid degrades gracefully to popularity
        candidates["collab_score"] = float(_model.trainset.global_mean)

    # --- Normalise & blend --------------------------------------------------
    candidates["collab_norm"] = (candidates["collab_score"] - 0.5) / 4.5

    candidates["hybrid_score"] = (
        popularity_weight * candidates["popularity_norm"]
        + collab_weight * candidates["collab_norm"]
    )

    return (
        candidates
        .sort_values("hybrid_score", ascending=False)
        [["title", "avg_rating", "num_ratings", "hybrid_score"]]
        .head(n)
        .reset_index(drop=True)
    )


# ===========================================================================
# MODULE INITIALISATION
# ===========================================================================

try:
    log.info("Initialising CineMatch recommender system…")
    _movies, _ratings, _tags = _load_data()
    _movie_stats, _top_movies = _build_popularity_model(_movies, _ratings)
    _content_movies, _tfidf_matrix, _indices = _build_content_model(_movies, _ratings, _tags)
    _model, _movie_title_map, _raw_iids = _load_collab_model(_movies)
    _hybrid_movies = _build_hybrid_frame(_movies, _movie_stats)
    log.info("System ready — all engines online.")
except FileNotFoundError as _err:
    log.warning("Module initialisation warning: %s", _err)
