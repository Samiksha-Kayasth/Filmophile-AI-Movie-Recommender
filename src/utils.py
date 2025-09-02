from sqlalchemy.exc import SQLAlchemyError
from src.db import SessionLocal, Feedback, Interaction
import json

# -----------------------------
# Feedback system (DB-backed)
# -----------------------------
def log_user_feedback(user_id: int, movie_id: int, feedback: str):
    """
    Log feedback ("Like" / "Dislike") for a given user and movie.
    Stored in PostgreSQL Feedback + Interaction tables.
    """
    with SessionLocal() as session:
        try:
            # Check if feedback already exists
            fb = session.query(Feedback).filter(
                Feedback.user_id == user_id,
                Feedback.movie_id == movie_id
            ).first()

            if not fb:
                fb = Feedback(
                    user_id=user_id,
                    movie_id=movie_id,
                    liked=True if feedback == "Like" else False
                )
                session.add(fb)
            else:
                # update existing
                fb.liked = True if feedback == "Like" else False

            # Also log in interactions
            inter = Interaction(
                user_id=user_id,
                movie_id=movie_id,
                event="like" if feedback == "Like" else "dislike",
                value=1.0 if feedback == "Like" else 0.0,
                context=json.dumps({"source": "streamlit"})
            )
            session.add(inter)

            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            raise RuntimeError(f"DB error logging feedback: {e}")

def get_user_preferences(user_id: int):
    """
    Retrieve stored preferences (likes, dislikes) for a user from DB.
    Returns dict with keys: {"likes": [movie_ids], "dislikes": [movie_ids]}.
    """
    with SessionLocal() as session:
        likes = session.query(Feedback.movie_id).filter(
            Feedback.user_id == user_id,
            Feedback.liked == True
        ).all()

        dislikes = session.query(Feedback.movie_id).filter(
            Feedback.user_id == user_id,
            Feedback.liked == False
        ).all()

        return {
            "likes": [m[0] for m in likes],
            "dislikes": [m[0] for m in dislikes]
        }
