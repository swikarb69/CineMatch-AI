"""
app.py
======
CineMatch — Streamlit entry point.

Loads CSS once from assets/style.css, renders the four recommendation
engines, and integrates with the TMDB API for posters and metadata.
"""

import html
import itertools
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from recommender import (
    get_top_movies,
    hybrid_recommend,
    recommend_content,
    recommend_for_user,
    search_movie,
)

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
LOGO_PATH = _ROOT / "assets" / "logo.png"
CSS_PATH = _ROOT / "assets" / "style.css"
LOGO = str(LOGO_PATH) if LOGO_PATH.exists() else None

st.set_page_config(
    page_title="CineMatch | Cozy Cinema",
    page_icon=LOGO if LOGO else "🍿",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()
TMDB_API_KEY: str | None = os.getenv("TMDB_API_KEY")

# ---------------------------------------------------------------------------
# CSS — injected ONCE per server session (cached by st.cache_data)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_css() -> str:
    """Read the stylesheet from disk; return empty string if missing."""
    if CSS_PATH.exists():
        return CSS_PATH.read_text(encoding="utf-8")
    return ""


def _inject_css() -> None:
    css = _load_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


_inject_css()

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
_STATE_DEFAULTS: dict = {
    "recommendations": None,
    "rec_type": None,
}
for _key, _val in _STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# Card UID counter
_card_counter = itertools.count(1)


# ===========================================================================
# TMDB API Helpers
# ===========================================================================

def _tmdb_available() -> bool:
    return bool(TMDB_API_KEY)


@st.cache_data(show_spinner=False, ttl=3600)
def get_movie_details(name: str) -> dict | None:
    """
    Fetch the first TMDB search result for `name`.
    Returns None when the API key is absent, the request fails, or no
    results are found.
    """
    if not _tmdb_available() or not name.strip():
        return None
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_API_KEY, "query": name.strip()},
            timeout=5,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        m = results[0]
        p = m.get("poster_path")
        b = m.get("backdrop_path")
        return {
            "id":       m.get("id"),
            "poster":   f"https://image.tmdb.org/t/p/w500{p}" if p else None,
            "backdrop": f"https://image.tmdb.org/t/p/w780{b}" if b else None,
            "overview": m.get("overview") or "",
            "rating":   m.get("vote_average") or 0,
            "votes":    m.get("vote_count") or 0,
            "release":  m.get("release_date") or "",
        }
    except (requests.RequestException, ValueError, KeyError):
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def get_genres(movie_id: int | None) -> list[str]:
    """Return a list of genre name strings for the given TMDB movie_id."""
    if not movie_id or not _tmdb_available():
        return []
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=5,
        )
        r.raise_for_status()
        return [g["name"] for g in r.json().get("genres", [])]
    except (requests.RequestException, ValueError, KeyError):
        return []


# ===========================================================================
# Dialog Modal
# ===========================================================================

@st.dialog(" ", width="large")
def show_movie_dialog(title: str, extra: str = "", index: int | None = None) -> None:
    """Native Streamlit dialog — renders movie detail card with TMDB data."""
    safe_title = html.escape(title)

    d = get_movie_details(title)
    if d and any([d["poster"], d["backdrop"], d["overview"], d["rating"]]):
        rating   = f"{d['rating']:.1f}" if d["rating"] else "N/A"
        year     = d["release"][:4] if d["release"] else "—"
        votes    = f"{d['votes']:,}" if d["votes"] else "—"
        overview = html.escape(d["overview"] or "No description available for this title.")
        genres   = get_genres(d["id"])
        tmdb_url = f"https://www.themoviedb.org/movie/{d['id']}" if d["id"] else ""
        img_src  = d["backdrop"] or d["poster"] or ""
    else:
        rating = "N/A"; year = "—"; votes = "—"
        overview = "No description available for this title."
        genres = []; tmdb_url = ""; img_src = ""

    rank_html   = f'<div class="cm-rank">#{index}</div>' if index else ""
    genre_pills = "".join(
        f'<span class="meta-pill pill-genre">{html.escape(g)}</span>' for g in genres[:5]
    )
    extra_pill  = f'<span class="meta-pill pill-extra">{html.escape(extra)}</span>' if extra else ""
    tmdb_btn    = (
        f'<a class="cm-tmdb" href="{tmdb_url}" target="_blank" rel="noopener noreferrer">'
        f'🎬 &nbsp;View on TMDB</a>'
    ) if tmdb_url else ""
    banner_html = (
        f'<img class="cm-backdrop" src="{img_src}" alt="{safe_title} backdrop">'
        if img_src else
        '<div class="cm-no-backdrop">🎬</div>'
    )

    st.markdown(f"""
<div class="cm-dialog">
{banner_html}
<div class="cm-body">
{rank_html}
<h2 class="cm-title">{safe_title}</h2>
<div class="cm-pills">
  <span class="meta-pill pill-rating">⭐ {rating}</span>
  <span class="meta-pill pill-year">📅 {year}</span>
  {genre_pills}
  {extra_pill}
</div>
<p class="cm-overview">{overview}</p>
<div class="cm-stats">
  <div class="cm-stat"><div class="cm-stat-val">⭐ {rating}</div><div class="cm-stat-lbl">TMDB Score</div></div>
  <div class="cm-stat"><div class="cm-stat-val">{votes}</div><div class="cm-stat-lbl">Votes</div></div>
  <div class="cm-stat"><div class="cm-stat-val">{year}</div><div class="cm-stat-lbl">Released</div></div>
</div>
{tmdb_btn}
</div>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# Card Renderer
# ===========================================================================

_FALLBACK_POSTER = "https://placehold.co/300x450/0d0b14/4a4560?text=No+Poster"


def render_card(title: str, extra: str = "", index: int | None = None) -> None:
    """Render a movie result card and attach a 'View details' dialog button."""
    uid = next(_card_counter)

    safe_title = html.escape(title)
    safe_extra = html.escape(extra) if extra else ""

    d = get_movie_details(title)
    poster   = (d["poster"] if d and d["poster"] else _FALLBACK_POSTER)
    rating   = f"{d['rating']:.1f}" if d and d["rating"] else "N/A"
    year     = d["release"][:4] if d and d["release"] else "Unknown"
    raw_ov   = (d["overview"] if d and d["overview"] else "No description available.")
    overview = html.escape(raw_ov[:260] + "…" if len(raw_ov) > 260 else raw_ov)

    rank_html  = f'<div class="movie-rank">#{index}</div>' if index is not None else ""
    extra_pill = f'<span class="meta-pill pill-extra">{safe_extra}</span>' if extra else ""

    st.markdown(f"""
<div class="movie-card">
  <div class="movie-poster">
    <img src="{poster}" alt="{safe_title}"
         onerror="this.onerror=null;this.src='{_FALLBACK_POSTER}'">
  </div>
  <div class="movie-info">
    {rank_html}
    <h3 class="movie-title">{safe_title}</h3>
    <div class="movie-meta">
      <span class="meta-pill pill-rating">⭐ {rating}</span>
      <span class="meta-pill pill-year">📅 {year}</span>
      {extra_pill}
    </div>
    <p class="movie-overview">{overview}</p>
  </div>
</div>
""", unsafe_allow_html=True)

    if st.button("↗  View details", key=f"viewbtn-{uid}"):
        show_movie_dialog(title, extra, index)


# ===========================================================================
# Sidebar
# ===========================================================================

with st.sidebar:
    if LOGO:
        c1, c2 = st.columns([1, 2.2], gap="small")
        with c1:
            st.image(LOGO, width=64)
        with c2:
            st.markdown("""
<div style="padding:.25rem 0 0;">
  <div style="font-size:1.4rem;font-weight:900;font-family:'Playfair Display',serif;
    background:linear-gradient(90deg,#ffb5a7,#e0aaff);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;line-height:1;">CineMatch</div>
  <div style="font-size:.65rem;color:#475569;letter-spacing:.1em;
    text-transform:uppercase;font-weight:600;margin-top:4px;">Cozy AI Cinema</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style="padding:.5rem 0 1rem;">
  <div style="font-size:1.5rem;font-weight:900;font-family:'Playfair Display',serif;
    background:linear-gradient(90deg,#ffb5a7,#e0aaff);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;">CineMatch</div>
  <div style="font-size:.72rem;color:#475569;letter-spacing:.1em;
    text-transform:uppercase;font-weight:600;margin-top:2px;">Cozy AI Cinema</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    recommender_type = st.radio("Choose Your Engine:", [
        "🔥 Trending & Popular",
        "🎭 Match the Vibe (Content)",
        "👥 For You (Collaborative)",
        "🚀 The Hybrid Formula",
    ])

    st.divider()

    st.markdown("""
<div style="font-size:.68rem;color:#334155;text-transform:uppercase;
  letter-spacing:.12em;font-weight:700;margin-bottom:10px;">Developer</div>
<div class="dev-card">
  <h4>Swikar Bhattarai</h4>
  <p>bhattaraimail2me@gmail.com</p>
  <div class="dev-links">
    <a href="https://github.com/swikarb69" target="_blank" rel="noopener noreferrer">🐙 &nbsp;GitHub</a>
    <a href="https://www.linkedin.com/in/swikar-bhattarai-11178b240" target="_blank" rel="noopener noreferrer">💼 &nbsp;LinkedIn</a>
    <a href="https://folio-swikarb69.vercel.app" target="_blank" rel="noopener noreferrer">🌐 &nbsp;Portfolio</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ===========================================================================
# Hero
# ===========================================================================

st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">🍿 &nbsp; AI-Powered Discovery</div>
  <div class="hero-title">Welcome to CineMatch.</div>
  <div class="hero-sub">
    Dim the lights, grab some popcorn, and let the AI find your next favourite film.
    Powered by popularity, content similarity, collaborative filtering, and hybrid intelligence.
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Movies in Vault", "87,585")
c2.metric("User Ratings", "32M+")
c3.metric("Active Cinephiles", "200K+")
c4.metric("AI Models", "4")
st.divider()

# ===========================================================================
# Engines
# ===========================================================================

_card_counter = itertools.count(1)

# ---------------------------------------------------------------------------
# 🔥  Popularity
# ---------------------------------------------------------------------------
if recommender_type == "🔥 Trending & Popular":
    st.markdown('<div class="section-label">🔥 &nbsp; Most Loved</div>', unsafe_allow_html=True)
    st.subheader("Universally Adored Films")
    st.caption("The highest-rated, most-watched movies in the entire database.")

    top_n = st.slider("How many films to surface?", 5, 50, 10)
    if st.button("Show me the hits ✦", use_container_width=True):
        with st.spinner("Curating the best of cinema…"):
            try:
                st.session_state.recommendations = get_top_movies(top_n)
                st.session_state.rec_type = "popular"
            except Exception as e:
                st.error(f"Engine error: {e}")

    if st.session_state.rec_type == "popular" and st.session_state.recommendations is not None:
        recs = st.session_state.recommendations
        for i, (_, row) in enumerate(recs.iterrows(), 1):
            render_card(row["title"], f"👥 {int(row['num_ratings']):,} votes", index=i)
        st.download_button(
            "⬇ Download as CSV",
            recs.to_csv(index=False),
            file_name="popular_movies.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# 🎭  Content-based
# ---------------------------------------------------------------------------
elif recommender_type == "🎭 Match the Vibe (Content)":
    st.markdown('<div class="section-label">🎭 &nbsp; Content Match</div>', unsafe_allow_html=True)
    st.subheader("Find Your Vibe")
    st.caption("Enter a movie you love, and we'll find others with the exact same energy.")

    search_text = st.text_input(
        "Search for a movie:",
        placeholder="e.g. Interstellar, Parasite, The Notebook…",
    )
    if search_text:
        try:
            matches = search_movie(search_text)
            movie_list = matches["title"].tolist() if not matches.empty else []
            if movie_list:
                selected = st.selectbox("Did you mean one of these?", movie_list)
                if st.button("Find Similar Films ✦", use_container_width=True):
                    with st.spinner("Analysing cinematic vibes…"):
                        st.session_state.recommendations = recommend_content(selected)
                        st.session_state.rec_type = "content"
            else:
                st.markdown(
                    '<div style="text-align:center;padding:3rem;color:#334155;">'
                    "🎬 No films found. Try another title.</div>",
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"Search error: {e}")

    if st.session_state.rec_type == "content" and st.session_state.recommendations is not None:
        recs = st.session_state.recommendations
        if "Error" in recs.columns:
            st.warning(recs["Error"].iloc[0])
        else:
            for i, (_, row) in enumerate(recs.iterrows(), 1):
                render_card(row["title"], f"✨ {row['similarity'] * 100:.1f}% match", index=i)

# ---------------------------------------------------------------------------
# 👥  Collaborative
# ---------------------------------------------------------------------------
elif recommender_type == "👥 For You (Collaborative)":
    st.markdown('<div class="section-label">👥 &nbsp; Collaborative</div>', unsafe_allow_html=True)
    st.subheader("Hand-Picked For You")
    st.caption("Based on users with the exact same taste profile as you.")

    user_id = st.number_input("Enter your User ID", min_value=1, value=1, step=1)
    if st.button("Curate my list ✦", use_container_width=True):
        with st.spinner("Consulting the AI taste engine…"):
            try:
                st.session_state.recommendations = recommend_for_user(int(user_id))
                st.session_state.rec_type = "collab"
            except Exception as e:
                st.error(f"Collaborative engine error: {e}")

    if st.session_state.rec_type == "collab" and st.session_state.recommendations is not None:
        recs = st.session_state.recommendations
        for i, (_, row) in enumerate(recs.iterrows(), 1):
            render_card(row["title"], f"🎯 {row['predicted_rating']:.2f} predicted", index=i)

# ---------------------------------------------------------------------------
# 🚀  Hybrid
# ---------------------------------------------------------------------------
elif recommender_type == "🚀 The Hybrid Formula":
    st.markdown('<div class="section-label">🚀 &nbsp; Hybrid AI</div>', unsafe_allow_html=True)
    st.subheader("The Ultimate Blend")
    st.caption("Tailored to your taste, filtered for actual quality and cultural impact.")

    user_id = st.number_input("User ID", min_value=1, value=1, step=1, key="hybrid_user")
    if st.button("Run Hybrid Engine ✦", use_container_width=True):
        with st.spinner("Mixing collaborative & popularity signals…"):
            try:
                st.session_state.recommendations = hybrid_recommend(int(user_id))
                st.session_state.rec_type = "hybrid"
            except Exception as e:
                st.error(f"Hybrid engine error: {e}")

    if st.session_state.rec_type == "hybrid" and st.session_state.recommendations is not None:
        recs = st.session_state.recommendations
        for i, (_, row) in enumerate(recs.iterrows(), 1):
            render_card(row["title"], f"🔥 {row['hybrid_score']:.3f} score", index=i)

# ===========================================================================
# Footer
# ===========================================================================

st.markdown("""
<div class="footer">
  Built with ❤️ by <strong>Swikar Bhattarai</strong><br>
  CineMatch &nbsp;·&nbsp; MovieLens 32M &nbsp;·&nbsp; TMDB API
</div>
""", unsafe_allow_html=True)
