"""Authentication module with bcrypt hashing, rate limiting, and password migration."""
import json
import os
import secrets
import time

import logging

import bcrypt

logger = logging.getLogger(__name__)

AUTH_FILE = "auth.json"

# --- Rate limiting ---
_login_attempts: dict[str, list[float]] = {}  # ip -> [timestamps]
MAX_ATTEMPTS = 5
COOLDOWN_SECONDS = 60


def _hash_password(password: str) -> str:
    """Hash a password with bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_bcrypt(password: str, stored: str) -> bool:
    """Verify a password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode(), stored.encode())
    except Exception:
        return False


def _verify_legacy_sha256(password: str, stored: str) -> bool:
    """Verify against old SHA-256 format (salt:hash) for migration."""
    import hashlib
    if ":" not in stored or stored.startswith("$2"):
        return False
    salt, hashed = stored.split(":", 1)
    check = hashlib.sha256((salt + password).encode()).hexdigest()
    return check == hashed


def _is_bcrypt_hash(stored: str) -> bool:
    """Check if a stored hash is bcrypt format."""
    return stored.startswith("$2")


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
        logger.info("Auth: Created default admin user (admin/admin)")
    else:
        logger.info(f"Auth: Loaded {len(auth['users'])} user(s)")


def check_rate_limit(ip: str) -> tuple[bool, int]:
    """Check if an IP is rate limited. Returns (allowed, seconds_remaining)."""
    now = time.time()
    attempts = _login_attempts.get(ip, [])
    # Remove old attempts outside the cooldown window
    attempts = [t for t in attempts if now - t < COOLDOWN_SECONDS]
    _login_attempts[ip] = attempts

    if len(attempts) >= MAX_ATTEMPTS:
        oldest = attempts[0]
        remaining = int(COOLDOWN_SECONDS - (now - oldest))
        return False, max(remaining, 1)

    return True, 0


def record_failed_attempt(ip: str) -> None:
    """Record a failed login attempt for an IP."""
    if ip not in _login_attempts:
        _login_attempts[ip] = []
    _login_attempts[ip].append(time.time())


def clear_attempts(ip: str) -> None:
    """Clear failed attempts for an IP after successful login."""
    _login_attempts.pop(ip, None)


def verify_login(username: str, password: str) -> bool:
    """Check username/password. Auto-migrates SHA-256 to bcrypt."""
    auth = _load_auth()
    users = auth.get("users", {})
    user = users.get(username)
    if not user:
        return False

    stored = user["password"]

    # Try bcrypt first
    if _is_bcrypt_hash(stored):
        return _verify_bcrypt(password, stored)

    # Try legacy SHA-256 and migrate if valid
    if _verify_legacy_sha256(password, stored):
        # Migrate to bcrypt
        user["password"] = _hash_password(password)
        _save_auth(auth)
        logger.info(f"Auth: Migrated '{username}' password from SHA-256 to bcrypt")
        return True

    return False


def change_password(username: str, old_password: str, new_password: str) -> tuple[bool, str]:
    """Change a user's password. Returns (success, message)."""
    if not new_password or len(new_password) < 4:
        return False, "Password must be at least 4 characters"

    auth = _load_auth()
    users = auth.get("users", {})
    user = users.get(username)

    if not user:
        return False, "User not found"

    # Verify old password (supports both formats)
    stored = user["password"]
    valid = False
    if _is_bcrypt_hash(stored):
        valid = _verify_bcrypt(old_password, stored)
    else:
        valid = _verify_legacy_sha256(old_password, stored)

    if not valid:
        return False, "Current password is incorrect"

    user["password"] = _hash_password(new_password)
    _save_auth(auth)
    return True, "Password changed successfully"


def generate_session_token() -> str:
    """Generate a random session token."""
    return secrets.token_hex(32)


# Session store (in-memory, with TTL)
SESSION_TTL = 86400  # 24 hours
_sessions: dict[str, tuple[str, float]] = {}  # token -> (username, created_at)


def create_session(username: str) -> str:
    """Create a session and return the token."""
    token = generate_session_token()
    _sessions[token] = (username, time.time())
    return token


def get_session_user(token: str) -> str | None:
    """Get the username for a session token, or None if expired."""
    entry = _sessions.get(token)
    if entry is None:
        return None
    username, created_at = entry
    if time.time() - created_at > SESSION_TTL:
        del _sessions[token]
        return None
    return username


def delete_session(token: str) -> None:
    """Delete a session."""
    _sessions.pop(token, None)


def cleanup_expired_sessions() -> int:
    """Remove expired sessions. Returns count removed."""
    now = time.time()
    expired = [t for t, (_, created) in _sessions.items() if now - created > SESSION_TTL]
    for t in expired:
        del _sessions[t]
    return len(expired)
