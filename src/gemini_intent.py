import json
from typing import Dict, Any
import google.generativeai as genai
from config import GEMINI_API_KEY

SYSTEM_PROMPT = """
You parse free-form user queries about movies into a compact JSON intent.

Return ONLY valid JSON with keys:
- "mode": one of ["mood", "similar", "filter", "open"]
- "mood": optional string like happy, sad, thrilled, scared, nostalgic, chill
- "similar_title": optional string (movie name if user asked for 'like X')
- "keywords": optional list of words/phrases (e.g., 'time travel', 'heist')
- "year_min": optional integer
- "year_max": optional integer

Examples:
User: 'i feel sad want something heartfelt' -> {"mode":"mood","mood":"sad"}
User: 'movies like Inception' -> {"mode":"similar","similar_title":"Inception"}
User: 'find a sci-fi heist 2010-2015' -> {"mode":"filter","keywords":["sci-fi","heist"],"year_min":2010,"year_max":2015}
User: 'surprise me' -> {"mode":"open"}
"""

def _configure():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")

def parse_intent(query: str) -> Dict[str, Any]:
    """
    Uses Gemini to convert a natural language query into a structured intent JSON.
    """
    model = _configure()
    prompt = f"{SYSTEM_PROMPT}\nUser: {query}\nJSON:"
    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()
    # Make a best-effort to find JSON in the reply
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        data = json.loads(text)
    except Exception:
        data = {"mode": "open", "keywords": [query]}
    return data
