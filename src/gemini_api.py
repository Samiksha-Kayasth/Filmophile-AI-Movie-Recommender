import google.generativeai as genai
import os
import json
import re

# Load API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("⚠️ GEMINI_API_KEY not found! Did you create a .env file?")

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")  # use the stable supported model


def _safe_json_parse(text: str):
    """
    Try to safely parse Gemini output as JSON.
    Falls back to wrapping plain text in a dict.
    """
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return [{"title": text.strip(), "year": None, "reason": "AI suggestion"}]

def gemini_recommend(user_query: str, liked_movies=None, top_k: int = 7) -> list:
    """
    Ask Gemini for movie recommendations.
    Returns a list of dicts with {title, year, reason}.
    """
    liked_str = ", ".join(liked_movies) if liked_movies else "None"

    prompt = f"""
    You are a movie recommendation system.
    The user asked: "{user_query}".
    They liked these movies: {liked_str}.

    Recommend {top_k} movies. 
    Format strictly as JSON array:
    [
      {{"title": "Movie Title", "year": 2000, "reason": "why recommended"}},
      ...
    ]
    """

    try:
        response = model.generate_content(prompt)
        recs = _safe_json_parse(response.text.strip())

        clean_recs = []
        for r in recs:
            clean_recs.append({
                "title": r.get("title") if isinstance(r, dict) else str(r),
                "year": r.get("year") if isinstance(r, dict) else None,
                "reason": r.get("reason") if isinstance(r, dict) else "AI suggestion"
            })
        return clean_recs[:top_k]
    except Exception as e:
        return [{"title": "Could not fetch Gemini recs", "year": None, "reason": str(e)}]

# 👇 Wrapper class for compatibility with streamlit_app.py
class GeminiClient:
    def get_recommendations(self, user_query: str, liked_movies=None, top_k: int = 7):
        """
        Wrapper to match the interface expected in streamlit_app.py
        """
        return gemini_recommend(user_query, liked_movies, top_k)
