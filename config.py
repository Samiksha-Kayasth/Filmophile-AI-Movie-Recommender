# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# ==========================
# Database Configuration
# ==========================
DATABASE_URL = "postgresql+psycopg2://filmouser:filmophile@localhost:5432/filmophile"

 
    # fallback if no .env is present


# ==========================
# Streamlit App Metadata
# ==========================
APP_TITLE = "🎬 Filmophile - AI Movie Recommender"
APP_DESCRIPTION = """
Welcome to **Filmophile**!  
An AI-powered movie recommendation system that uses **Gemini** to analyze your mood, 
interests, and past activity to suggest the best movies for you.
"""

# ==========================
# Authentication Settings
# ==========================
MIN_PASS_LEN = 6
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "supersecretkey")  # replace in production
JWT_ALGORITHM = "HS256"

# ==========================
# Gemini / AI Integration
# ==========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key")

# ==========================
# Miscellaneous
# ==========================
EVAL_THRESHOLD = 0.6  # threshold for adjusting recommendations

# config.py (add at bottom or under Miscellaneous)

from pathlib import Path

# ==========================
# Data Paths
# ==========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"   # all datasets go inside Filmoplile/data/
MODEL_DIR = BASE_DIR / "models"

# Ensure folders exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# config.py

# Path where user preferences will be saved
USER_PREFS_FILE = "data/user_prefs.json"
