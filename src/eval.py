import json
import numpy as np
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
from src.db import SessionLocal, Interaction, UserVector
from datetime import datetime

class TasteUpdater:
    """
    Online updates to user's taste vector using interactions.
    Positive feedback pulls vector toward the movie, negative pushes away.
    """
    def __init__(self, vectorizer, movie_ids, X):
        self.vectorizer = vectorizer
        self.movie_ids = movie_ids
        self.X = X

    def _get_uv(self, user_id: int) -> np.ndarray:
        with SessionLocal() as s:
            uv = s.query(UserVector).filter(UserVector.user_id==user_id).first()
            if not uv:
                return np.zeros(self.X.shape[1], dtype=np.float32)
            data = json.loads(uv.vector_json)
            v = np.zeros(self.X.shape[1], dtype=np.float32)
            for k, val in data.items():
                k = int(k)
                if k < v.shape[0]:
                    v[k] = float(val)
            return v

    def _save_uv(self, user_id: int, v: np.ndarray):
        nz = v.nonzero()[0]
        payload = {int(i): float(v[i]) for i in nz}
        with SessionLocal() as s:
            uv = s.query(UserVector).filter(UserVector.user_id==user_id).first()
            if uv:
                uv.vector_json = json.dumps(payload)
            else:
                uv = UserVector(user_id=user_id, vector_json=json.dumps(payload))
                s.add(uv)
            s.commit()

    def update_from_event(self, user_id: int, movie_id: int, event: str, value: Optional[float] = None):
        """
        event: like | dislike | rate
        value: rating 0..5 for 'rate'
        """
        # learning rates
        alpha_like = 0.6
        alpha_dislike = 0.4
        alpha_rate = 0.25

        # fetch movie vector
        idx = np.where(self.movie_ids == movie_id)[0]
        if len(idx) == 0:
            return
        mv = self.X[idx[0]].toarray().ravel().astype(np.float32)

        u = self._get_uv(user_id)
        if event == "like":
            u = u + alpha_like * mv
        elif event == "dislike":
            u = u - alpha_dislike * mv
        elif event == "rate" and value is not None:
            # center around 3; scale to [-2, +2]
            delta = float(value) - 3.0
            u = u + alpha_rate * delta * mv

        # normalize to unit length to keep cosine meaningful
        norm = np.linalg.norm(u)
        if norm > 0:
            u = u / norm
        self._save_uv(user_id, u)

def log_interaction(user_id: int, movie_id: int, event: str, value=None, context: str | None = None):
    with SessionLocal() as s:
        s.add(Interaction(
            user_id=user_id,
            movie_id=movie_id,
            event=event,
            value=value,
            context=context
        ))
        s.commit()
