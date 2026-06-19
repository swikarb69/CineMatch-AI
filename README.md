# 🍿 CineMatch — Cozy AI Cinema

> *Dim the lights, grab some popcorn, and let the AI find your next favourite film.*

<br>

![Python](https://img.shields.io/badge/Python-3.11-a78bfa?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ffb5a7?style=flat-square&logo=streamlit&logoColor=white)
![MovieLens](https://img.shields.io/badge/Dataset-MovieLens%2032M-e0aaff?style=flat-square)
![TMDB](https://img.shields.io/badge/API-TMDB-c4b5fd?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-94a3b8?style=flat-square)

<br>

CineMatch is a full-stack, AI-powered movie recommendation web app built with Streamlit. It combines four distinct recommendation engines — popularity-based, content-based, collaborative filtering, and a hybrid model — all wrapped in a cozy midnight cinema UI with live TMDB poster art.

<br>

---

## ✨ Features

| Engine | What It Does |
|--------|-------------|
| 🔥 **Trending & Popular** | Bayesian-weighted scoring surfaces the most universally loved films |
| 🎭 **Match the Vibe** | TF-IDF content similarity on genres + user tags finds films with the same energy |
| 👥 **For You** | SVD collaborative filtering predicts ratings based on users with identical taste profiles |
| 🚀 **Hybrid Formula** | Blends collaborative (70%) and popularity (30%) signals into one definitive score |

- **87,585 movies** from the MovieLens 32M dataset
- **32M+ user ratings** powering the collaborative model
- **Live TMDB posters**, ratings, and overviews for every recommendation
- Beautiful custom dark UI with Playfair Display typography and animated movie cards
- CSV export for any recommendation list
- Graceful fallbacks — no poster? No problem.

<br>

---

## 🗂 Project Structure

```
ML-32M/
│
├── app.py                        # Streamlit frontend
├── recommender.py                # All 4 recommendation engines
│
├── data/
│   ├── movies.csv                # Movie titles & genres
│   ├── ratings.csv               # 32M user ratings
│   ├── tags.csv                  # User-generated tags
│   ├── links.csv                 # IMDB / TMDB ID mapping
│   └── checksums.txt
│
├── models/
│   └── svd_model.pkl             # Pre-trained SVD model (Surprise)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_popularity_recommender.ipynb
│   ├── 03_content_based_recommender.ipynb
│   ├── 04_collaborative_filtering.ipynb
│   └── 05_hybrid_recommender.ipynb
│
├── outputs/
│   ├── top_movies.csv
│   ├── content_based_movies.csv
│   ├── user1_recommendations.csv
│   └── hybrid_recommendations.csv
│
├── assets/
│   └── logo.png
│
├── .env                          # API keys (never commit this)
├── requirements.txt
└── .gitignore
```

<br>

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/swikarb69/cinematch.git
cd cinematch
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your environment variables

Create a `.env` file in the project root:

```env
TMDB_API_KEY=your_tmdb_api_key_here
```

Get a free API key at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api).

### 5. Download the MovieLens 32M dataset

Download from [grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/) and place the CSV files inside the `data/` folder.

```
data/movies.csv
data/ratings.csv
data/tags.csv
data/links.csv
```

### 6. Train the SVD model

Run the collaborative filtering notebook to generate `models/svd_model.pkl`:

```bash
jupyter notebook notebooks/04_collaborative_filtering.ipynb
```

Or run it headlessly:

```bash
jupyter nbconvert --to notebook --execute notebooks/04_collaborative_filtering.ipynb
```

### 7. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

<br>

---

## 🧠 How Each Engine Works

### 🔥 Popularity — Bayesian Weighted Score

Uses the same formula as IMDb's Top 250 to avoid favouring films with very few ratings:

```
score = (v / (v + m)) × R  +  (m / (v + m)) × C
```

Where `v` = vote count, `m` = 90th percentile vote threshold, `R` = average rating, `C` = global mean rating.

---

### 🎭 Content-Based — TF-IDF Cosine Similarity

Each movie is represented as a text document combining its **genres** and **user-generated tags**. TF-IDF vectors are computed and cosine similarity (via `linear_kernel`) finds the nearest neighbours. Results are filtered by a minimum rating count to suppress obscure matches.

---

### 👥 Collaborative Filtering — SVD Matrix Factorisation

Trained with the [Surprise](https://surpriselib.com/) library using **Singular Value Decomposition**. The model decomposes the user-item rating matrix into latent factor vectors, learning both user preferences (`pu`) and item characteristics (`qi`). For a given user it predicts ratings on all unseen movies and returns the top-N.

---

### 🚀 Hybrid — Weighted Score Fusion

Combines normalised collaborative scores and normalised popularity scores with a **70 / 30 split**:

```
hybrid_score = 0.7 × collab_norm  +  0.3 × popularity_norm
```

The collaborative score is computed via fast vectorised dot products (`qi @ pu`) rather than looping over predictions, making it significantly faster for large candidate sets.

<br>

---

## 📓 Notebooks

Each notebook is standalone and documents the full development process:

| Notebook | Contents |
|----------|----------|
| `01_data_exploration` | Dataset shape, missing values, rating distributions, genre analysis |
| `02_popularity_recommender` | Bayesian scoring, top-movie extraction, baseline evaluation |
| `03_content_based_recommender` | Tag cleaning, TF-IDF pipeline, similarity matrix, evaluation |
| `04_collaborative_filtering` | SVD hyperparameter tuning, cross-validation, model serialisation |
| `05_hybrid_recommender` | Score fusion experiments, weight optimisation, final evaluation |

<br>

---

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TMDB_API_KEY` | — | Required for poster images and metadata |
| `min_ratings` (content) | `50` | Minimum votes for a film to appear in content results |
| `n` (all engines) | `10` | Number of recommendations to return |
| Hybrid weights | `0.7 / 0.3` | Collaborative vs popularity blend ratio |

<br>

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| ML — Collaborative | Surprise (SVD) |
| ML — Content | scikit-learn (TF-IDF, cosine similarity) |
| Data | pandas, NumPy |
| Poster API | TMDB REST API |
| Environment | python-dotenv |
| Serialisation | joblib |

<br>

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

<br>

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Dataset: [MovieLens 32M](https://grouplens.org/datasets/movielens/) — F. Maxwell Harper and Joseph A. Konstan, 2015. *The MovieLens Datasets: History and Context.*

<br>

---

## 👨‍💻 Developer

**Swikar Bhattarai**

[![GitHub](https://img.shields.io/badge/GitHub-swikarb69-e0aaff?style=flat-square&logo=github)](https://github.com/swikarb69)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-swikar--bhattarai-c4b5fd?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/swikar-bhattarai-11178b240)
[![Portfolio](https://img.shields.io/badge/Portfolio-portfolio--swikarb69.vercel.app-ffb5a7?style=flat-square)](https://portfolio-swikarb69.vercel.app)

<br>

---

<div align="center">
  <sub>Built with ❤️ and way too much popcorn &nbsp;·&nbsp; CineMatch &nbsp;·&nbsp; MovieLens 32M &nbsp;·&nbsp; TMDB API</sub>
</div>