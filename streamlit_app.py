# streamlit_app.py

import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from sqlalchemy.exc import IntegrityError
from src.recommender import Recommender
from src.auth import register_user, login_user, get_current_user, logout_user
from src.db import SessionLocal, Feedback
from src.utils import get_user_preferences
from config import APP_TITLE
from src.gemini_api import gemini_recommend  



# Load .env once for the whole app
load_dotenv()
from src.db import init_db
init_db()  # ensures tables exist

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("🎥 Filmophile – Your AI Movie Recommender")

# -----------------------------
# Singletons (cached across reruns)
# -----------------------------
if "rec_engine" not in st.session_state:
    st.session_state["rec_engine"] = Recommender()
rec_engine: Recommender = st.session_state["rec_engine"]

# -----------------------------
# Session Init
# -----------------------------
if "user" not in st.session_state:
    st.session_state["user"] = None
if "page" not in st.session_state:
    st.session_state["page"] = "login"

# -----------------------------
# Helpers
# -----------------------------
def resolve_movie_id(rec_engine: Recommender, movie_id, title: str):
    if movie_id is not None:
        try:
            return int(movie_id)
        except Exception:
            pass
    try:
        matches = rec_engine.lookup[rec_engine.lookup["title"].str.lower() == str(title).lower()]
        if not matches.empty:
            return int(matches.index[0])
    except Exception:
        pass
    try:
        matches = rec_engine.lookup[rec_engine.lookup["title"].str.contains(str(title), case=False, na=False)]
        if not matches.empty:
            return int(matches.index[0])
    except Exception:
        pass
    return int(hash(title) % 100000)

def save_feedback_and_update(user_id: int, movie_id: int, liked: bool, rec_engine: Recommender):
    with SessionLocal() as s:
        try:
            fb = s.query(Feedback).filter(Feedback.user_id == user_id, Feedback.movie_id == movie_id).first()
            if fb:
                fb.liked = liked
            else:
                fb = Feedback(user_id=user_id, movie_id=movie_id, liked=liked)
                s.add(fb)
            s.commit()
        except Exception as e:
            s.rollback()
            st.error(f"Could not save feedback: {e}")
            return
    try:
        rec_engine.log_interaction(user_id=user_id, movie_id=movie_id, liked=liked)
    except Exception as e:
        st.warning(f"Saved feedback but failed to update recommender vectors: {e}")
    st.success("Feedback saved ✅")

# -----------------------------
# UI helpers
# -----------------------------
def _render_movie_row(user_id: int, rec_engine: Recommender, movie_id, title: str, prefix="🎬", source="default"):
    resolved_id = resolve_movie_id(rec_engine, movie_id, title)
    unique_key = f"{source}_{resolved_id}_{abs(hash(title))}"  # ✅ ensures uniqueness

    col1, col2, col3 = st.columns([6, 1, 1])
    with col1:
        st.write(f"{prefix} **{title}**")
    with col2:
        if st.button("👍", key=f"like_{unique_key}"):
            save_feedback_and_update(user_id, resolved_id, liked=True, rec_engine=rec_engine)
            st.rerun()
    with col3:
        if st.button("👎", key=f"dislike_{unique_key}"):
            save_feedback_and_update(user_id, resolved_id, liked=False, rec_engine=rec_engine)
            st.rerun()

def render_recs_list(user_id: int, rec_engine: Recommender, recs, prefix="🎬", source="default"):
    """
    Supports:
      - pandas.DataFrame with columns movieId, title
      - list[str] of titles
      - list[dict] with keys: title, year, reason
    """
    if recs is None:
        st.info("No recommendations to show.")
        return

    if isinstance(recs, pd.DataFrame):
        if recs.empty:
            st.info("No recommendations found.")
            return
        for _, row in recs.iterrows():
            movie_id = row.get("movieId")
            title = row.get("title", "Unknown Title")
            reason = row.get("reason", None)
            _render_movie_row(user_id, rec_engine, movie_id, title, prefix, source)
            if reason:
                st.caption(reason)
        return

    if isinstance(recs, (list, tuple)) and len(recs) > 0:
        if isinstance(recs[0], dict):  # Gemini style
            for r in recs:
                title = r.get("title", "Unknown Title")
                year = r.get("year")
                reason = r.get("reason")
                display = f"{title} ({year})" if year else title
                _render_movie_row(user_id, rec_engine, None, display, prefix, source)
                if reason:
                    st.caption(reason)
            return
        elif isinstance(recs[0], str):  # plain strings
            for title in recs:
                _render_movie_row(user_id, rec_engine, None, title, prefix, source)
            return

    st.error("Unexpected recommendation format.")

# -----------------------------
# Auth UI
# -----------------------------
def login_page():
    st.sidebar.subheader("🔑 Login / Register")
    tab_login, tab_register = st.sidebar.tabs(["Login", "Register"])
    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="login_btn"):
            ok, msg = login_user(email, password)
            if ok:
                st.success(msg)
                st.session_state["page"] = "main"
                st.rerun()
            else:
                st.error(msg)
    with tab_register:
        email_r = st.text_input("Email", key="reg_email")
        display_name_r = st.text_input("Display Name", key="reg_name")
        password_r = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Register", key="reg_btn"):
            ok, msg = register_user(email_r, password_r, display_name_r)
            if ok:
                st.success(msg)
                st.session_state["page"] = "main"
                st.rerun()
            else:
                st.error(msg)

# -----------------------------
# Main App
# -----------------------------
def main_page():
    user = get_current_user()
    if not user:
        st.warning("⚠️ Please log in first.")
        st.stop()

    st.sidebar.write(f"👋 Hello, **{user['display_name']}**")
    if st.sidebar.button("Logout"):
        logout_user()
        st.session_state["page"] = "login"
        st.rerun()

    st.subheader("🔍 Find something to watch")
    query = st.text_input("Type a mood, genre or movie title (e.g., 'thriller like Inception')", key="query_main")

    # Hybrid recommendations: TF-IDF + Gemini
    if st.button("✨ Get Recommendations", key="get_recs_btn"):
        try:
            user_q = query.strip() if query else None
            tfidf_recs = rec_engine.get_recommendations(user_query=user_q, user_id=user["id"], top_k=7)
        except Exception as e:
            st.error(f"TF-IDF Recommendation error: {e}")
            tfidf_recs = None

        try:
            gemini_recs = gemini_recommend(query or "good movies", top_k=7)
        except Exception as e:
            st.error(f"Gemini Recommendation error: {e}")
            gemini_recs = None
    else:
        tfidf_recs = rec_engine.get_recommendations(user_query=None, user_id=user["id"], top_k=7)
        gemini_recs = None

    # Show TF-IDF recommendations 📊
    st.header("📊 TF-IDF Recommendations")
    render_recs_list(user["id"], rec_engine, tfidf_recs, prefix="📊", source="tfidf")

    # Divider
    st.divider()

    # Show Gemini recommendations 🤖
    if gemini_recs:
        st.header("🤖 Gemini AI Suggestions")
        render_recs_list(user["id"], rec_engine, gemini_recs, prefix="🤖", source="gemini")

    # Divider
    st.divider()

    # Because you liked...
    with SessionLocal() as s:
        last_like = (
            s.query(Feedback)
            .filter(Feedback.user_id == user["id"], Feedback.liked == True)
            .order_by(Feedback.created_at.desc())
            .first()
        )
    if last_like:
        try:
            liked_title = rec_engine.lookup.loc[last_like.movie_id, "title"]
        except Exception:
            liked_title = f"Movie {last_like.movie_id}"
        st.subheader(f"💡 Because you liked **{liked_title}**")
        try:
            sim_df = rec_engine.similar_to(last_like.movie_id, top_k=20)
            sim_df = rec_engine.personalize(user["id"], sim_df)
            sim_df = rec_engine.top_n(sim_df, n=7)
            render_recs_list(user["id"], rec_engine, sim_df, prefix="✨", source="similar")
        except Exception:
            st.info("No similar recommendations available.")

    # Divider
    st.divider()

    # Taste snapshot
    st.subheader("📊 Your Taste Profile")
    prefs = get_user_preferences(user["id"])
    if prefs:
        st.json(prefs)
    else:
        st.info("No feedback yet. Click 👍 or 👎 to build your profile.")

# -----------------------------
# Router
# -----------------------------
if st.session_state["page"] == "login":
    login_page()
else:
    main_page()
