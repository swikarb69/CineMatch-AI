import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


print("Loading datasets...")

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
tags = pd.read_csv("data/tags.csv")


# ============================================================
# POPULARITY MODEL
# ============================================================

print("Building popularity model...")

movie_stats = ratings.groupby("movieId").agg(
    avg_rating=("rating", "mean"),
    num_ratings=("rating", "count")
).reset_index()

movie_stats = movie_stats.merge(
    movies[["movieId", "title"]],
    on="movieId"
)

C = movie_stats["avg_rating"].mean()
m = movie_stats["num_ratings"].quantile(0.90)

movie_stats["score"] = (
    (
        movie_stats["num_ratings"]
        /
        (movie_stats["num_ratings"] + m)
    )
    * movie_stats["avg_rating"]
) + (
    (
        m
        /
        (movie_stats["num_ratings"] + m)
    )
    * C
)

top_movies = movie_stats[
    movie_stats["num_ratings"] >= m
].copy()

top_movies = top_movies.sort_values(
    "score",
    ascending=False
)


def get_top_movies(n=10):

    return (
        top_movies[
            [
                "title",
                "avg_rating",
                "num_ratings",
                "score"
            ]
        ]
        .head(n)
        .reset_index(drop=True)
    )


# ============================================================
# CONTENT BASED MODEL
# ============================================================

print("Building content-based model...")

tags = tags.dropna(subset=["tag"])

movie_tags = (
    tags.groupby("movieId")["tag"]
    .apply(lambda x: " ".join(x.astype(str)))
    .reset_index()
)

content_movies = movies.merge(
    movie_tags,
    on="movieId",
    how="left"
)

content_movies["tag"] = content_movies["tag"].fillna("")

rating_stats = ratings.groupby("movieId").agg(
    avg_rating=("rating", "mean"),
    num_ratings=("rating", "count")
).reset_index()

content_movies = content_movies.merge(
    rating_stats,
    on="movieId",
    how="left"
)

content_movies["avg_rating"] = (
    content_movies["avg_rating"]
    .fillna(0)
)

content_movies["num_ratings"] = (
    content_movies["num_ratings"]
    .fillna(0)
    .astype(int)
)

content_movies["content"] = (
    content_movies["genres"]
    + " "
    + content_movies["tag"]
)

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    content_movies["content"]
)

indices = pd.Series(
    content_movies.index,
    index=content_movies["title"]
)

indices = indices[
    ~indices.index.duplicated(keep="first")
]


def search_movie(query):

    return content_movies[
        content_movies["title"]
        .str.contains(
            query,
            case=False,
            na=False
        )
    ][["title"]].head(20)


def recommend_content(
    movie_title,
    n=10,
    min_ratings=50
):

    if movie_title not in indices:

        return pd.DataFrame(
            {"Error": [f"{movie_title} not found"]}
        )

    idx = indices[movie_title]

    sim_scores = linear_kernel(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    sim_indices = sim_scores.argsort()[
        -(n * 5 + 1):
    ][::-1]

    sim_indices = sim_indices[
        sim_indices != idx
    ]

    candidates = content_movies.iloc[
        sim_indices
    ][
        [
            "title",
            "genres",
            "avg_rating",
            "num_ratings"
        ]
    ].copy()

    candidates["similarity"] = sim_scores[
        sim_indices
    ]

    candidates = candidates[
        candidates["num_ratings"]
        >= min_ratings
    ]

    return (
        candidates
        .sort_values(
            ["similarity", "avg_rating"],
            ascending=False
        )
        .head(n)
        .reset_index(drop=True)
    )


# ============================================================
# COLLABORATIVE FILTERING
# ============================================================

print("Loading trained SVD model...")

model = joblib.load(
    "models/svd_model.pkl"
)

print("SVD model loaded successfully!")

movie_title_map = (
    movies.set_index("movieId")["title"]
    .to_dict()
)


def recommend_for_user(
    user_id,
    n=10
):

    rated_movies = set(
        ratings[
            ratings["userId"] == user_id
        ]["movieId"]
    )

    all_movies = set(
        movies["movieId"]
    )

    unrated_movies = list(
        all_movies - rated_movies
    )

    predictions = []

    for movie_id in unrated_movies[:5000]:

        pred = model.predict(
            user_id,
            movie_id
        )

        predictions.append(
            (
                movie_id,
                pred.est
            )
        )

    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for movie_id, score in predictions[:n]:

        recommendations.append(
            {
                "title":
                movie_title_map.get(
                    movie_id,
                    "Unknown"
                ),
                "predicted_rating":
                round(score, 2)
            }
        )

    return pd.DataFrame(
        recommendations
    )


# ============================================================
# HYBRID MODEL
# ============================================================

print("Building hybrid model...")

hybrid_movies = movies.merge(
    movie_stats[
        [
            "movieId",
            "avg_rating",
            "num_ratings",
            "score"
        ]
    ],
    on="movieId",
    how="left"
)

hybrid_movies["score"] = (
    hybrid_movies["score"]
    .fillna(0)
)

hybrid_movies["avg_rating"] = (
    hybrid_movies["avg_rating"]
    .fillna(0)
)

hybrid_movies["num_ratings"] = (
    hybrid_movies["num_ratings"]
    .fillna(0)
)

hybrid_movies["popularity_norm"] = (
    hybrid_movies["score"]
    /
    hybrid_movies["score"].max()
)


def hybrid_recommend(
    user_id,
    n=10
):

    rated_movies = set(
        ratings[
            ratings["userId"] == user_id
        ]["movieId"]
    )

    candidate_movies = hybrid_movies[
        ~hybrid_movies["movieId"]
        .isin(rated_movies)
    ].copy()

    try:

        inner_uid = (
            model.trainset
            .to_inner_uid(user_id)
        )

        mu = model.trainset.global_mean

        bu = model.bu[inner_uid]

        pu = model.pu[inner_uid]

        all_scores = np.clip(
            mu + bu + model.bi +
            model.qi @ pu,
            0.5,
            5.0
        )

        raw_iids = np.array(
            [
                int(
                    model.trainset
                    .to_raw_iid(i)
                )
                for i in range(
                    model.trainset.n_items
                )
            ]
        )

        score_series = pd.Series(
            all_scores,
            index=raw_iids
        )

        fallback = float(
            np.clip(
                mu + bu,
                0.5,
                5.0
            )
        )

        candidate_movies[
            "collab_score"
        ] = (
            candidate_movies["movieId"]
            .map(score_series)
            .fillna(fallback)
        )

    except ValueError:

        candidate_movies[
            "collab_score"
        ] = model.trainset.global_mean

    candidate_movies["collab_norm"] = (
        (
            candidate_movies["collab_score"]
            - 0.5
        )
        / 4.5
    )

    candidate_movies["hybrid_score"] = (
        0.3
        * candidate_movies["popularity_norm"]
        +
        0.7
        * candidate_movies["collab_norm"]
    )

    return (
        candidate_movies
        .sort_values(
            "hybrid_score",
            ascending=False
        )
        [
            [
                "title",
                "avg_rating",
                "num_ratings",
                "hybrid_score"
            ]
        ]
        .head(n)
        .reset_index(drop=True)
    )


print("System Ready!")