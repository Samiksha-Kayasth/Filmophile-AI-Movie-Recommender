import bcrypt
from sqlalchemy.exc import IntegrityError
from src.db import SessionLocal, User
import streamlit as st

MIN_PASS_LEN = 6

# ---------------------------
# Password helpers
# ---------------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ---------------------------
# Auth core
# ---------------------------
def register_user(email: str, password: str, display_name: str):
    """
    Register a new user.
    Returns (success: bool, message: str).
    """
    if len(password) < MIN_PASS_LEN:
        return False, f"Password must be at least {MIN_PASS_LEN} characters"

    with SessionLocal() as s:
        # check if email already exists
        existing = s.query(User).filter(User.email == email.lower()).first()
        if existing:
            return False, "Email already registered"

        # create new user
        user = User(
            email=email.lower(),
            password_hash=hash_password(password),
            display_name=display_name
        )
        s.add(user)
        try:
            s.commit()
            s.refresh(user)

            # ✅ Auto-login after registration
            st.session_state["user"] = {
                "id": user.id,
                "email": user.email,
                "display_name": user.display_name,
            }

            return True, f"User {display_name} registered successfully and logged in!"
        except IntegrityError:
            s.rollback()
            return False, "Database error while registering user"


def login_user(email: str, password: str):
    """
    Validate credentials and set session if success.
    Returns (success: bool, message: str).
    """
    with SessionLocal() as s:
        user = s.query(User).filter(User.email == email.lower()).first()
        if not user:
            return False, "Email not found"
        if not check_password(password, user.password_hash):
            return False, "Invalid password"

        # ✅ Save in Streamlit session
        st.session_state["user"] = {
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
        }
        return True, f"Welcome back, {user.display_name}!"


def get_current_user():
    """Return the current logged-in user from session."""
    return st.session_state.get("user", None)


def logout_user():
    """Clear the session user."""
    st.session_state["user"] = None
