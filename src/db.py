from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, Text, ForeignKey, Boolean
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func
from config import DATABASE_URL   # ✅ Load from config.py
import os

# ---------------------------
# Database Config
# ---------------------------
Base = declarative_base()

# ✅ Use SQLite safely on Streamlit Cloud
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        future=True,
        connect_args={"check_same_thread": False}  # required for SQLite
    )
else:
    engine = create_engine(DATABASE_URL, echo=False, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# ---------------------------
# Models
# ---------------------------

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    feedback = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")
    interactions = relationship("Interaction", back_populates="user", cascade="all, delete-orphan")
    vector = relationship("UserVector", back_populates="user", uselist=False, cascade="all, delete-orphan")


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    movie_id = Column(Integer, index=True, nullable=False)
    liked = Column(Boolean, nullable=False)  # True = 👍, False = 👎
    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="feedback")


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    movie_id = Column(Integer, index=True, nullable=True)
    event = Column(String(50))                 # view | like | dislike | rate | search | recommend_click
    value = Column(Float, nullable=True)       # optional rating or weight
    context = Column(Text, nullable=True)      # JSON string (intent, mood, query)
    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="interactions")


class UserVector(Base):
    __tablename__ = "user_vectors"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    vector_json = Column(Text, nullable=False)  # store JSON string
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="vector")


# ---------------------------
# Utility
# ---------------------------

def init_db():
    """Initialize database tables."""
    # 👇 This ensures all models are loaded before table creation
    import src.db   # replace with src.models if models are in another file
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized (tables created).")
