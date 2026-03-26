"""Simple authentication module with hashed passwords."""
import hashlib
import json
import os
import secrets
from pathlib import Path

AUTH_FILE = "auth.json"


def _hash_password(password: str, salt: str = "") -> str:
    """Hash a password with SHA-256 + salt."""
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def _verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored hash."""
    if ":" not in stored:
        return False
    salt, hashed = stored.split(":", 1)
    check = hashlib.sha256((salt + password).encode()).hexdigest()
    return check == hashed


def _load_auth() -> dict:
    """Load auth data from file."""
    if os.path.exists(AUTH_FILE):
        try:
            with open(AUTH_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_auth(data: dict) -> None:
    """Save auth data to file."""
    with open(AUTH_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def initialize_auth() -> None:
    """Create default admin user if no auth file exists."""
    auth = _load_auth()
    if not auth.get("users"):
        auth["users"] = {
            "admin": {
                "password": _hash_password("admin"),
                "role": "admin"
            }
        }
        _save_auth(auth)
        print("Auth: Created default admin user (admin/admin)")
    else:
        print(f"Auth: Loaded {len(auth['users'])} user(s)")


def verify_login(username: str, password: str) -> bool:
    """Check username/password against stored credentials."""
    auth = _load_auth()
    users = auth.get("users", {})
    user = users.get(username)
    if not user:
        return False
    return _verify_password(password, user["password"])


def change_password(username: str, old_password: str, new_password: str) -> tuple[bool, str]:
    """Change a user's password. Returns (success, message)."""
    if not new_password or len(new_password) < 4:
        return False, "Password must be at least 4 characters"

    auth = _load_auth()
    users = auth.get("users", {})
    user = users.get(username)

    if not user:
        return False, "User not found"

    if not _verify_password(old_password, user["password"]):
        return False, "Current password is incorrect"

    user["password"] = _hash_password(new_password)
    _save_auth(auth)
    return True, "Password changed successfully"


def generate_session_token() -> str:
    """Generate a random session token."""
    return secrets.token_hex(32)


# Session store (in-memory, simple)
_sessions: dict[str, str] = {}  # token -> username


def create_session(username: str) -> str:
    """Create a session and return the token."""
    token = generate_session_token()
    _sessions[token] = username
    return token


def get_session_user(token: str) -> str | None:
    """Get the username for a session token, or None if invalid."""
    return _sessions.get(token)


def delete_session(token: str) -> None:
    """Delete a session."""
    _sessions.pop(token, None)
