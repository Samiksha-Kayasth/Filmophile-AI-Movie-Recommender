import json
import random
import numpy as np
import pandas as pd
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data_prep import DataStore
from src.db import SessionLocal, Interaction, UserVector, Feedback
from src.gemini_api import gemini_recommend  # <-- make sure this exists

# Mood → genres mapping
MOOD_TO_GENRES = {
    "happy": ["Comedy", "Animation", "Family", "Romance"],
    "sad": ["Drama"],
    "thrilled": ["Thriller", "Action", "Mystery"],
    "scared": ["Horror"],
    "nostalgic": ["Adventure", "Fantasy"],
    "chill": ["Drama", "Romance", "Comedy"],
}


class Recommender:
    def __init__(self):
        # Load movie metadata
        self.store = DataStore()
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        texts = self.store.get_movie_text()
        self.movie_ids = texts["movieId"].values
        self.X = self.vectorizer.fit_transform(texts["text"].values)

        self.lookup = self.store.movie_lookup().set_index("movieId")
        pop = self.store.ratings.groupby("movieId")["rating"].count().rename("pop")
        self.pop = pop

        # Cold start backup
        self.all_movies = self.lookup["title"].tolist()

    # ---------- User Vector ----------
    def _get_user_vector(self, user_id: int) -> np.ndarray:
        with SessionLocal() as s:
            uv = s.query(UserVector).filter(UserVector.user_id == user_id).first()
            if uv is None:
                return np.zeros(self.X.shape[1], dtype=np.float32)
            data = json.loads(uv.vector_json)
            v = np.zeros(self.X.shape[1], dtype=np.float32)
            for k, val in data.items():
                k = int(k)
                if k < v.shape[0]:
                    v[k] = float(val)
            return v

    def _save_user_vector(self, user_id: int, v: np.ndarray):
        nz = v.nonzero()[0]
        payload = {int(i): float(v[i]) for i in nz}
        with SessionLocal() as s:
            uv = s.query(UserVector).filter(UserVector.user_id == user_id).first()
            if uv:
                uv.vector_json = json.dumps(payload)
            else:
                s.add(UserVector(user_id=user_id, vector_json=json.dumps(payload)))
            s.commit()

    # ---------- Interaction Logging ----------
    def log_interaction(self, user_id: int, movie_id: int, liked: bool):
        with SessionLocal() as s:
            s.add(Interaction(user_id=user_id, movie_id=movie_id,
                              event="like" if liked else "dislike",
                              value=1.0 if liked else -1.0))
            s.commit()

        # Update user vector
        idx = np.where(self.movie_ids == movie_id)[0]
        if len(idx) == 0:
            return
        idx = idx[0]
        movie_vec = self.X[idx].toarray().ravel()
        user_vec = self._get_user_vector(user_id)
        user_vec = user_vec + movie_vec if liked else user_vec - movie_vec
        self._save_user_vector(user_id, user_vec)

    # ---------- TF-IDF / Popularity ----------
    def similar_to(self, movie_id: int, top_k: int = 20) -> pd.DataFrame:
        idx = np.where(self.movie_ids == movie_id)[0][0]
        sims = cosine_similarity(self.X[idx], self.X).ravel()
        order = sims.argsort()[::-1]

        recs = []
        for j in order[: top_k + 1]:
            mid = int(self.movie_ids[j])
            if mid == movie_id:
                continue
            title = self.lookup.loc[mid, "title"]
            genres = self.lookup.loc[mid, "genres"]
            recs.append((mid, title, genres, float(sims[j])))
        return pd.DataFrame(recs, columns=["movieId", "title", "genres", "score"])

    def by_keywords(self, keywords: Union[str, List[str]], top_k: int = 50) -> pd.DataFrame:
        if not keywords:
            return self.by_popular(top_k)
        if isinstance(keywords, list):
            q = " ".join(keywords)
        else:
            q = str(keywords)
        qv = self.vectorizer.transform([q])
        sims = cosine_similarity(qv, self.X).ravel()
        order = sims.argsort()[::-1][:top_k]
        rows = []
        for j in order:
            mid = int(self.movie_ids[j])
            title = self.lookup.loc[mid, "title"]
            genres = self.lookup.loc[mid, "genres"]
            rows.append((mid, title, genres, float(sims[j])))
        return pd.DataFrame(rows, columns=["movieId", "title", "genres", "score"])

    def by_genres(self, genres: List[str], top_k: int = 50) -> pd.DataFrame:
        mask = self.lookup["genres"].apply(lambda g: any(gen in g for gen in genres))
        df = self.lookup[mask].reset_index().copy()
        df["score"] = 1.0
        return df[["movieId", "title", "genres", "score"]].head(top_k)

    def by_popular(self, top_k: int = 50) -> pd.DataFrame:
        base = self.lookup.copy().join(self.pop, how="left").fillna({"pop": 0})
        base = base.sort_values("pop", ascending=False)
        base["score"] = base["pop"]
        return base.reset_index()[["movieId", "title", "genres", "score"]].head(top_k)

    # ---------- Personalization ----------
    def personalize(self, user_id: int, candidates: pd.DataFrame, alpha: float = 0.7) -> pd.DataFrame:
        if candidates.empty:
            return candidates
        u = self._get_user_vector(user_id)
        if np.allclose(u, 0):
            candidates["pScore"] = 0.0
        else:
            idxs = [np.where(self.movie_ids == mid)[0][0] for mid in candidates["movieId"].values if mid in self.lookup.index]
            if not idxs:
                candidates["pScore"] = 0.0
            else:
                Xcand = self.X[idxs]
                sims = cosine_similarity(u.reshape(1, -1), Xcand).ravel()
                candidates = candidates.copy()
                candidates["pScore"] = sims
        candidates["final"] = (
            (1 - alpha) * candidates["score"].rank(method="dense", ascending=False) / len(candidates)
            + alpha * candidates["pScore"]
        )
        return candidates.sort_values("final", ascending=False).reset_index(drop=True)

    # ---------- Main Wrapper ----------
    def get_recommendations(self, user_query=None, user_id=None, top_k=8):
        results = []

        # --- Gemini Recommendations ---
        liked_titles = []
        if user_id:
            with SessionLocal() as s:
                feedbacks = s.query(Feedback).filter(Feedback.user_id == user_id, Feedback.liked == True).all()
                liked_titles = [f"Movie {f.movie_id}" for f in feedbacks]

        try:
            gemini_recs = gemini_recommend(user_query or "Suggest movies", liked_titles, top_k=top_k//2)
            for rec in gemini_recs:
                rec["source"] = "🔮 Gemini"
            results.extend(gemini_recs)
        except Exception as e:
            print(f"[WARN] Gemini failed: {e}")

        # --- TF-IDF / Popularity Recommendations ---
        if isinstance(user_query, str) and user_query.strip():
            q = user_query.lower().strip()
            if q in MOOD_TO_GENRES:
                recs = self.by_genres(MOOD_TO_GENRES[q], top_k=top_k)
            elif not self.lookup[self.lookup["title"].str.contains(q, case=False, na=False)].empty:
                recs = self.similar_to_title(q, top_k=top_k)
            else:
                recs = self.by_keywords(q, top_k=top_k)
        else:
            recs = self.by_popular(top_k=top_k)

        if user_id is not None and not recs.empty:
            recs = self.personalize(user_id, recs)

        for _, row in recs.head(top_k//2).iterrows():
            results.append({
                "title": row["title"],
                "year": None,
                "reason": "TF-IDF / Popularity",
                "source": "📝 TF-IDF"
            })

        return results

    def similar_to_title(self, title: str, top_k: int = 20):
        matches = self.lookup[self.lookup["title"].str.contains(title, case=False, na=False)]
        if matches.empty:
            return pd.DataFrame(columns=["movieId", "title", "genres", "score"])
        target_id = int(matches.index[0])
        return self.similar_to(target_id, top_k=top_k)
