import streamlit as st
import requests
import os
from pathlib import Path
from dotenv import load_dotenv
from recommender import (
    get_top_movies, search_movie,
    recommend_content, recommend_for_user, hybrid_recommend
)

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"
LOGO = str(LOGO_PATH) if LOGO_PATH.exists() else None

st.set_page_config(
    page_title="CineMatch | Cozy Cinema",
    page_icon=LOGO if LOGO else "🍿", layout="wide",
    initial_sidebar_state="expanded"
)
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
for key, default in [
    ("selected_movie", None),
    ("recommendations", None),
    ("rec_type", None),
    ("close_modal", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------------------------------------
# CSS
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800;900&family=Playfair+Display:wght@700;900&display=swap');
html,body,[class*="css"]{font-family:'Nunito',sans-serif;}

.stApp{
  background-color:#09070f;
  background-image:radial-gradient(ellipse 100% 50% at 50% -10%,rgba(157,78,221,.13) 0%,transparent 70%);
  color:#e2e8f0;
}

/* sidebar */
section[data-testid="stSidebar"]{background:rgba(10,8,18,.97)!important;border-right:1px solid rgba(255,255,255,.04);}
section[data-testid="stSidebar"] .stRadio>label{color:#94a3b8!important;font-size:.75rem;letter-spacing:.1em;text-transform:uppercase;font-weight:700;}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label{background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.05);border-radius:12px;padding:10px 14px;margin-bottom:6px;transition:all .2s;font-size:.9rem;color:#cbd5e1;}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover{background:rgba(224,170,255,.07);border-color:rgba(224,170,255,.25);color:#e0aaff;}

/* hero */
.hero{padding:3.5rem 3rem;border-radius:28px;border:1px solid rgba(255,255,255,.06);background:linear-gradient(135deg,rgba(157,78,221,.12) 0%,rgba(15,12,28,.9) 50%,rgba(255,166,158,.07) 100%);margin-bottom:2rem;}
.hero-eyebrow{font-size:.72rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:rgba(224,170,255,.6);margin-bottom:10px;}
.hero-title{font-family:'Playfair Display',serif;font-size:4rem;font-weight:900;background:linear-gradient(100deg,#ffb5a7 0%,#e0aaff 50%,#a8d8ff 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 .75rem;line-height:1.1;letter-spacing:-1.5px;}
.hero-sub{font-size:1.05rem;font-weight:300;color:#8898a9;line-height:1.7;max-width:620px;}

/* metrics */
div[data-testid="metric-container"]{background:rgba(255,255,255,.025)!important;border:1px solid rgba(255,255,255,.06)!important;border-radius:16px!important;padding:1rem 1.25rem!important;}
div[data-testid="metric-container"] label{color:#64748b!important;font-size:.72rem!important;letter-spacing:.1em!important;text-transform:uppercase!important;}
div[data-testid="metric-container"] div[data-testid="stMetricValue"]{color:#e2e8f0!important;font-weight:700!important;font-size:1.6rem!important;}

h2,.stSubheader{font-family:'Playfair Display',serif!important;font-size:1.6rem!important;font-weight:700!important;color:#f1f5f9!important;}

/* main buttons */
.stButton>button{background:linear-gradient(135deg,rgba(157,78,221,.25),rgba(255,166,158,.15))!important;border:1px solid rgba(224,170,255,.25)!important;color:#e0aaff!important;border-radius:14px!important;font-weight:700!important;font-family:'Nunito',sans-serif!important;padding:.6rem 1.5rem!important;transition:all .25s ease!important;font-size:.95rem!important;}
.stButton>button:hover{background:linear-gradient(135deg,rgba(157,78,221,.4),rgba(255,166,158,.25))!important;border-color:rgba(224,170,255,.5)!important;transform:translateY(-2px)!important;box-shadow:0 6px 20px rgba(157,78,221,.2)!important;}

/* dialog inner */
.cm-dialog{margin:-1rem -1rem 0;}
.cm-dialog .cm-backdrop,.cm-dialog .cm-no-backdrop{border-radius:12px 12px 0 0;}

/* inputs */
.stTextInput>div>div>input{background:rgba(255,255,255,.04)!important;border:1px solid rgba(255,255,255,.08)!important;border-radius:12px!important;color:#e2e8f0!important;}
.stTextInput>div>div>input:focus{border-color:rgba(224,170,255,.4)!important;box-shadow:0 0 0 3px rgba(157,78,221,.1)!important;}
.stSelectbox>div>div{background:rgba(255,255,255,.04)!important;border:1px solid rgba(255,255,255,.08)!important;border-radius:12px!important;color:#e2e8f0!important;}

hr{border:none!important;border-top:1px solid rgba(255,255,255,.06)!important;margin:1.5rem 0!important;}

/* cards */
.movie-card{display:flex;background:rgba(255,255,255,.025);border-radius:22px;padding:22px;margin-bottom:14px;border:1px solid rgba(255,255,255,.05);border-left:3px solid transparent;gap:24px;transition:all .28s cubic-bezier(.22,1,.36,1);cursor:pointer;}
.movie-card:hover{transform:translateY(-4px);background:rgba(255,255,255,.045);border-color:rgba(224,170,255,.22);border-left-color:#c4b5fd;box-shadow:0 16px 40px rgba(0,0,0,.45);}
.movie-card:hover .click-hint{opacity:1!important;}
.movie-poster img{border-radius:14px;width:130px;height:195px;object-fit:cover;display:block;box-shadow:0 8px 24px rgba(0,0,0,.6);transition:transform .3s ease;flex-shrink:0;}
.movie-card:hover .movie-poster img{transform:scale(1.04);}
.movie-info{display:flex;flex-direction:column;justify-content:center;flex:1;min-width:0;}
.movie-rank{font-size:.68rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:rgba(224,170,255,.45);margin-bottom:5px;}
.movie-title{font-family:'Playfair Display',serif;font-size:1.45rem;font-weight:700;color:#f8fafc;margin:0 0 10px;line-height:1.25;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.movie-meta{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;align-items:center;}
.meta-pill{display:inline-flex;align-items:center;gap:5px;padding:4px 11px;border-radius:20px;font-size:.78rem;font-weight:700;}
.pill-rating{background:rgba(245,197,24,.12);color:#f5c518;border:1px solid rgba(245,197,24,.2);}
.pill-year{background:rgba(148,163,184,.08);color:#94a3b8;border:1px solid rgba(148,163,184,.12);}
.pill-extra{background:rgba(224,170,255,.1);color:#c4b5fd;border:1px solid rgba(224,170,255,.18);}
.pill-genre{background:rgba(99,179,237,.08);color:#63b3ed;border:1px solid rgba(99,179,237,.15);font-size:.72rem;}
.movie-overview{font-size:.88rem;color:#64748b;line-height:1.65;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;}
.click-hint{margin-top:10px;font-size:.72rem;color:rgba(196,181,253,.5);letter-spacing:.06em;opacity:0;transition:opacity .2s ease;}

.section-label{display:inline-flex;align-items:center;gap:8px;background:rgba(224,170,255,.07);border:1px solid rgba(224,170,255,.12);border-radius:20px;padding:5px 14px;font-size:.75rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:rgba(224,170,255,.7);margin-bottom:.75rem;width:fit-content;}

.dev-card{background:rgba(255,255,255,.025);padding:16px;border-radius:16px;border:1px solid rgba(255,255,255,.05);}
.dev-card h4{margin:0 0 2px;color:#f1f5f9;font-size:.95rem;font-weight:700;}
.dev-card p{color:#475569;font-size:.8rem;margin:0 0 12px;}
.dev-links{display:flex;flex-direction:column;gap:6px;}
.dev-links a{color:#a78bfa;text-decoration:none;font-size:.85rem;font-weight:600;display:flex;align-items:center;gap:8px;padding:7px 10px;border-radius:10px;border:1px solid rgba(255,255,255,.04);background:rgba(255,255,255,.02);transition:all .2s;}
.dev-links a:hover{background:rgba(167,139,250,.08);border-color:rgba(167,139,250,.2);color:#c4b5fd;}

/* ── MODAL ── */
.cm-overlay{
  position:fixed;top:0;left:0;right:0;bottom:0;
  background:rgba(5,3,12,.88);
  z-index:99999;
  display:flex;align-items:center;justify-content:center;
  padding:24px;
  animation:cmFade .18s ease;
}
@keyframes cmFade{from{opacity:0}to{opacity:1}}
.cm-box{
  background:#110e1c;
  border:1px solid rgba(224,170,255,.18);
  border-radius:28px;max-width:700px;width:100%;
  max-height:90vh;overflow-y:auto;
  animation:cmSlide .22s cubic-bezier(.22,1,.36,1);
  scrollbar-width:thin;scrollbar-color:rgba(196,181,253,.15) transparent;
  position:relative;
}
@keyframes cmSlide{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:translateY(0)}}
.cm-backdrop{width:100%;height:230px;object-fit:cover;border-radius:26px 26px 0 0;display:block;}
.cm-no-backdrop{width:100%;height:230px;background:linear-gradient(135deg,rgba(157,78,221,.18),#0d0b18);border-radius:26px 26px 0 0;display:flex;align-items:center;justify-content:center;font-size:4rem;}
.cm-body{padding:28px 32px 32px;}
/* close button — top-right corner of the box */
.cm-close{
  position:absolute;top:16px;right:16px;
  background:rgba(20,17,35,.9);
  border:1px solid rgba(255,255,255,.15);
  border-radius:50%;width:40px;height:40px;
  cursor:pointer;color:#cbd5e1;font-size:1.1rem;
  display:flex;align-items:center;justify-content:center;
  transition:all .2s;z-index:10;
  line-height:1;
}
.cm-close:hover{background:rgba(224,170,255,.15);color:#e0aaff;border-color:rgba(224,170,255,.4);}
.cm-rank{font-size:.68rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:rgba(224,170,255,.4);margin-bottom:6px;}
.cm-title{font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:900;color:#f8fafc;margin:0 0 14px;line-height:1.2;}
.cm-pills{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;}
.cm-overview{font-size:.97rem;color:#8898a9;line-height:1.78;margin-bottom:26px;}
.cm-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:26px;}
.cm-stat{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:14px;text-align:center;}
.cm-stat-val{font-size:1.35rem;font-weight:800;color:#e2e8f0;}
.cm-stat-lbl{font-size:.68rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-top:3px;}
.cm-tmdb{display:inline-flex;align-items:center;gap:8px;background:rgba(1,180,228,.1);border:1px solid rgba(1,180,228,.25);color:#01b4e4;border-radius:12px;padding:10px 20px;text-decoration:none;font-weight:700;font-size:.88rem;transition:all .2s;}
.cm-tmdb:hover{background:rgba(1,180,228,.2);border-color:rgba(1,180,228,.45);}

.stDownloadButton>button{background:transparent!important;border:1px solid rgba(255,255,255,.08)!important;color:#64748b!important;border-radius:10px!important;font-size:.85rem!important;}
.stDownloadButton>button:hover{border-color:rgba(224,170,255,.2)!important;color:#a78bfa!important;}
.stSpinner>div{border-top-color:#9d4edd!important;}
.footer{text-align:center;color:#1e293b;padding:48px 20px 32px;font-size:.85rem;line-height:1.8;}
.footer strong{color:#334155;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TMDB HELPERS
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def get_movie_details(name):
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_API_KEY, "query": name}, timeout=6
        )
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
            "rating":   m.get("vote_average", 0),
            "votes":    m.get("vote_count", 0),
            "release":  m.get("release_date") or "",
        }
    except:
        return None

@st.cache_data(show_spinner=False)
def get_genres(movie_id):
    if not movie_id:
        return []
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY}, timeout=6
        )
        return [g["name"] for g in r.json().get("genres", [])]
    except:
        return []

# --------------------------------------------------
# MODAL  — native st.dialog (works reliably, real X button)
# --------------------------------------------------

@st.dialog(" ", width="large")
def show_movie_dialog(title, extra="", index=None):
    d = get_movie_details(title)

    if d and (d["poster"] or d["backdrop"] or d["overview"] or d["rating"]):
        rating   = f"{d['rating']:.1f}" if d["rating"] else "N/A"
        year     = d["release"][:4] if d["release"] else "—"
        votes    = f"{d['votes']:,}" if d["votes"] else "—"
        overview = d["overview"] if d["overview"] else "No description available for this title."
        genres   = get_genres(d["id"])
        tmdb_url = f"https://www.themoviedb.org/movie/{d['id']}" if d["id"] else ""
        img_src  = d["backdrop"] or d["poster"] or ""
    else:
        rating = "N/A"; year = "—"; votes = "—"
        overview = "No description available for this title."
        genres = []; tmdb_url = ""; img_src = ""

    rank_html   = f'<div class="cm-rank">#{index}</div>' if index else ""
    genre_pills = "".join(f'<span class="meta-pill pill-genre">{g}</span>' for g in genres[:5])
    extra_pill  = f'<span class="meta-pill pill-extra">{extra}</span>' if extra else ""
    tmdb_btn    = f'<a class="cm-tmdb" href="{tmdb_url}" target="_blank">🎬 &nbsp;View on TMDB</a>' if tmdb_url else ""

    if img_src:
        banner_html = f'<img class="cm-backdrop" src="{img_src}">'
    else:
        banner_html = '<div class="cm-no-backdrop">🎬</div>'

    st.markdown(f"""
<div class="cm-dialog">
{banner_html}
<div class="cm-body">
{rank_html}
<h2 class="cm-title">{title}</h2>
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

# --------------------------------------------------
# CARD RENDERER
# --------------------------------------------------

_uid = {"n": 0}

def render_card(title, extra="", index=None):
    _uid["n"] += 1
    uid = _uid["n"]

    d = get_movie_details(title)
    fallback = "https://placehold.co/300x450/0d0b14/4a4560?text=No+Poster"
    poster   = (d["poster"] if d and d["poster"] else fallback)
    rating   = f"{d['rating']:.1f}" if d and d["rating"] else "N/A"
    year     = d["release"][:4] if d and d["release"] else "Unknown"
    raw_ov   = d["overview"] if d and d["overview"] else "No description available."
    overview = raw_ov[:260] + "…" if len(raw_ov) > 260 else raw_ov

    rank_html  = f'<div class="movie-rank">#{index}</div>' if index is not None else ""
    extra_pill = f'<span class="meta-pill pill-extra">{extra}</span>' if extra else ""

    st.markdown(f"""
    <div class="movie-card">
      <div class="movie-poster">
        <img src="{poster}" alt="{title}" onerror="this.onerror=null;this.src='{fallback}'">
      </div>
      <div class="movie-info">
        {rank_html}
        <h3 class="movie-title">{title}</h3>
        <div class="movie-meta">
          <span class="meta-pill pill-rating">⭐ {rating}</span>
          <span class="meta-pill pill-year">📅 {year}</span>
          {extra_pill}
        </div>
        <p class="movie-overview">{overview}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Real Streamlit button — reliably opens the native dialog
    if st.button(f"↗  View details", key=f"viewbtn-{uid}"):
        show_movie_dialog(title, extra, index)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

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
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1;">
                  CineMatch</div>
                <div style="font-size:.65rem;color:#475569;letter-spacing:.1em;
                  text-transform:uppercase;font-weight:600;margin-top:4px;">Cozy AI Cinema</div>
              </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
          <div style="padding:.5rem 0 1rem;">
            <div style="font-size:1.5rem;font-weight:900;font-family:'Playfair Display',serif;
              background:linear-gradient(90deg,#ffb5a7,#e0aaff);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
              CineMatch</div>
            <div style="font-size:.72rem;color:#475569;letter-spacing:.1em;
              text-transform:uppercase;font-weight:600;margin-top:2px;">Cozy AI Cinema</div>
          </div>
        """, unsafe_allow_html=True)
    st.divider()
    recommender_type = st.radio("Choose Your Engine:", [
        "🔥 Trending & Popular",
        "🎭 Match the Vibe (Content)",
        "👥 For You (Collaborative)",
        "🚀 The Hybrid Formula"
    ])
    st.divider()
    st.markdown("""
      <div style="font-size:.68rem;color:#334155;text-transform:uppercase;letter-spacing:.12em;font-weight:700;margin-bottom:10px;">Developer</div>
      <div class="dev-card">
        <h4>Swikar Bhattarai</h4><p>bhattaraimail2me@gmail.com</p>
        <div class="dev-links">
          <a href="https://github.com/swikarb69" target="_blank">🐙 &nbsp;GitHub</a>
          <a href="https://www.linkedin.com/in/swikar-bhattarai-11178b240" target="_blank">💼 &nbsp;LinkedIn</a>
          <a href="https://folio-swikarb69.vercel.app" target="_blank">🌐 &nbsp;Portfolio</a>
        </div>
      </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# HERO
# --------------------------------------------------

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

# --------------------------------------------------
# ENGINES
# --------------------------------------------------

_uid["n"] = 0  # reset card counter each render

if recommender_type == "🔥 Trending & Popular":
    st.markdown('<div class="section-label">🔥 &nbsp; Most Loved</div>', unsafe_allow_html=True)
    st.subheader("Universally Adored Films")
    st.caption("The highest-rated, most-watched movies in the entire database.")
    top_n = st.slider("How many films to surface?", 5, 50, 10)
    if st.button("Show me the hits ✦", use_container_width=True):
        with st.spinner("Curating the best of cinema…"):
            st.session_state.recommendations = get_top_movies(top_n)
            st.session_state.rec_type = "popular"
    if st.session_state.rec_type == "popular" and st.session_state.recommendations is not None:
        recs = st.session_state.recommendations
        for i, (_, row) in enumerate(recs.iterrows(), 1):
            render_card(row["title"], f"👥 {int(row['num_ratings']):,} votes", index=i)
        st.download_button("⬇ Download as CSV", recs.to_csv(index=False), "popular_movies.csv")

elif recommender_type == "🎭 Match the Vibe (Content)":
    st.markdown('<div class="section-label">🎭 &nbsp; Content Match</div>', unsafe_allow_html=True)
    st.subheader("Find Your Vibe")
    st.caption("Enter a movie you love, and we'll find others with the exact same energy.")
    search_text = st.text_input("Search for a movie:", placeholder="e.g. Interstellar, Parasite, The Notebook…")
    if search_text:
        matches = search_movie(search_text)
        movie_list = matches["title"].tolist()
        if movie_list:
            selected = st.selectbox("Did you mean one of these?", movie_list)
            if st.button("Find Similar Films ✦", use_container_width=True):
                with st.spinner("Analysing cinematic vibes…"):
                    st.session_state.recommendations = recommend_content(selected)
                    st.session_state.rec_type = "content"
        else:
            st.markdown('<div style="text-align:center;padding:3rem;color:#334155;">🎬 No films found. Try another title.</div>', unsafe_allow_html=True)
    if st.session_state.rec_type == "content" and st.session_state.recommendations is not None:
        recs = st.session_state.recommendations
        for i, (_, row) in enumerate(recs.iterrows(), 1):
            render_card(row["title"], f"✨ {row['similarity']*100:.1f}% match", index=i)

elif recommender_type == "👥 For You (Collaborative)":
    st.markdown('<div class="section-label">👥 &nbsp; Collaborative</div>', unsafe_allow_html=True)
    st.subheader("Hand-Picked For You")
    st.caption("Based on users with the exact same taste profile as you.")
    user_id = st.number_input("Enter your User ID", min_value=1, value=1)
    if st.button("Curate my list ✦", use_container_width=True):
        with st.spinner("Consulting the AI taste engine…"):
            st.session_state.recommendations = recommend_for_user(int(user_id))
            st.session_state.rec_type = "collab"
    if st.session_state.rec_type == "collab" and st.session_state.recommendations is not None:
        recs = st.session_state.recommendations
        for i, (_, row) in enumerate(recs.iterrows(), 1):
            render_card(row["title"], f"🎯 {row['predicted_rating']:.2f} predicted", index=i)

elif recommender_type == "🚀 The Hybrid Formula":
    st.markdown('<div class="section-label">🚀 &nbsp; Hybrid AI</div>', unsafe_allow_html=True)
    st.subheader("The Ultimate Blend")
    st.caption("Tailored to your taste, filtered for actual quality and cultural impact.")
    user_id = st.number_input("User ID", min_value=1, value=1, key="hybrid_user")
    if st.button("Run Hybrid Engine ✦", use_container_width=True):
        with st.spinner("Mixing collaborative & popularity signals…"):
            st.session_state.recommendations = hybrid_recommend(int(user_id))
            st.session_state.rec_type = "hybrid"
    if st.session_state.rec_type == "hybrid" and st.session_state.recommendations is not None:
        recs = st.session_state.recommendations
        for i, (_, row) in enumerate(recs.iterrows(), 1):
            render_card(row["title"], f"🔥 {row['hybrid_score']:.3f} score", index=i)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
  Built with ❤️ by <strong>Swikar Bhattarai</strong><br>
  CineMatch &nbsp;·&nbsp; MovieLens 32M &nbsp;·&nbsp; TMDB API
</div>
""", unsafe_allow_html=True)