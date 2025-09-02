import os
import pandas as pd
from config import DATA_DIR

MOVIES = os.path.join(DATA_DIR, "movies.csv")
RATINGS = os.path.join(DATA_DIR, "ratings.csv")
TAGS = os.path.join(DATA_DIR, "tags.csv")
LINKS = os.path.join(DATA_DIR, "links.csv")

class DataStore:
    def __init__(self):
        needed = [MOVIES, RATINGS, TAGS]
        if not all(os.path.exists(p) for p in needed):
            raise FileNotFoundError(
                f"MovieLens CSVs not found in {DATA_DIR}. "
                "Place movies.csv, ratings.csv, tags.csv, links.csv."
            )
        self.movies = pd.read_csv(MOVIES)
        self.ratings = pd.read_csv(RATINGS)
        self.tags = pd.read_csv(TAGS)
        self.links = pd.read_csv(LINKS) if os.path.exists(LINKS) else pd.DataFrame()
        self._prepare()

    def _prepare(self):
        tags_agg = self.tags.groupby("movieId")["tag"].apply(lambda x: " ".join(map(str, x))).reset_index()
        self.movies = self.movies.merge(tags_agg, on="movieId", how="left")
        self.movies["tag"] = self.movies["tag"].fillna("")
        self.movies["text"] = (
            self.movies["title"].fillna("") + " " +
            self.movies["genres"].fillna("") + " " +
            self.movies["tag"].fillna("")
        ).str.replace("|", " ", regex=False)

    def movie_lookup(self):
        return self.movies[["movieId", "title", "genres"]]

    def get_movie_text(self):
        return self.movies[["movieId", "text"]]
