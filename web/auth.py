"""Authentication module with bcrypt hashing, rate limiting, RBAC, and password migration."""
import json
import os
import secrets
import time

import logging

import bcrypt

logger = logging.getLogger(__name__)

import os as _os
_CONFIG_DIR = _os.path.join("data", "config") if _os.path.isdir(_os.path.join("data", "config")) else "."
AUTH_FILE = _os.path.join(_CONFIG_DIR, "auth.json")
VALID_ROLES = ("admin", "operator", "viewer")
ROLE_HIERARCHY = {"admin": 3, "operator": 2, "viewer": 1}

# --- Rate limiting ---
_login_attempts: dict[str, list[float]] = {}
MAX_ATTEMPTS = 5
COOLDOWN_SECONDS = 60


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_bcrypt(password: str, stored: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), stored.encode())
    except Exception:
        return False


def _verify_legacy_sha256(password: str, stored: str) -> bool:
    import hashlib
    if ":" not in stored or stored.startswith("$2"):
        return False
    salt, hashed = stored.split(":", 1)
    check = hashlib.sha256((salt + password).encode()).hexdigest()
    return check == hashed


def _is_bcrypt_hash(stored: str) -> bool:
    return stored.startswith("$2")


def _load_auth() -> dict:
    if os.path.exists(AUTH_FILE):
        try:
            with open(AUTH_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_auth(data: dict) -> None:
    with open(AUTH_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def initialize_auth() -> None:
    auth = _load_auth()
    if not auth.get("users"):
        auth["users"] = {
            "admin": {"password": _hash_password("admin"), "role": "admin"}
        }
        _save_auth(auth)
        logger.info("Auth: Created default admin user (admin/admin)")
    else:
        logger.info(f"Auth: Loaded {len(auth['users'])} user(s)")


# --- Rate limiting ---

def check_rate_limit(ip: str) -> tuple[bool, int]:
    now = time.time()
    attempts = _login_attempts.get(ip, [])
    attempts = [t for t in attempts if now - t < COOLDOWN_SECONDS]
    _login_attempts[ip] = attempts
    if len(attempts) >= MAX_ATTEMPTS:
        remaining = int(COOLDOWN_SECONDS - (now - attempts[0]))
        return False, max(remaining, 1)
    return True, 0


def record_failed_attempt(ip: str) -> None:
    if ip not in _login_attempts:
        _login_attempts[ip] = []
    _login_attempts[ip].append(time.time())


def clear_attempts(ip: str) -> None:
    _login_attempts.pop(ip, None)


# --- Login & Password ---

def verify_login(username: str, password: str) -> bool:
    auth = _load_auth()
    users = auth.get("users", {})
    user = users.get(username)
    if not user:
        return False

    stored = user["password"]
    if _is_bcrypt_hash(stored):
        return _verify_bcrypt(password, stored)

    if _verify_legacy_sha256(password, stored):
        user["password"] = _hash_password(password)
        _save_auth(auth)
        logger.info(f"Auth: Migrated '{username}' password to bcrypt")
        return True
    return False


def reset_password(username: str, new_password: str) -> tuple[bool, str]:
    """Admin reset — no old password required."""
    if not new_password or len(new_password) < 4:
        return False, "Password must be at least 4 characters"
    auth = _load_auth()
    user = auth.get("users", {}).get(username)
    if not user:
        return False, "User not found"
    user["password"] = _hash_password(new_password)
    _save_auth(auth)
    logger.info(f"Auth: Admin reset password for '{username}'")
    return True, f"Password reset for '{username}'"


def change_password(username: str, old_password: str, new_password: str) -> tuple[bool, str]:
    if not new_password or len(new_password) < 4:
        return False, "Password must be at least 4 characters"

    auth = _load_auth()
    user = auth.get("users", {}).get(username)
    if not user:
        return False, "User not found"

    stored = user["password"]
    valid = _verify_bcrypt(old_password, stored) if _is_bcrypt_hash(stored) else _verify_legacy_sha256(old_password, stored)
    if not valid:
        return False, "Current password is incorrect"

    user["password"] = _hash_password(new_password)
    _save_auth(auth)
    return True, "Password changed successfully"


# --- User Management (RBAC) ---

def get_user_role(username: str) -> str:
    auth = _load_auth()
    user = auth.get("users", {}).get(username)
    return user.get("role", "viewer") if user else "viewer"


def get_all_users() -> list[dict]:
    auth = _load_auth()
    return [{"username": name, "role": u.get("role", "viewer")}
            for name, u in auth.get("users", {}).items()]


def create_user(username: str, password: str, role: str) -> tuple[bool, str]:
    username = username.strip()
    if not username or len(username) < 2:
        return False, "Username must be at least 2 characters"
    if not password or len(password) < 4:
        return False, "Password must be at least 4 characters"
    if role not in VALID_ROLES:
        return False, f"Invalid role. Must be one of: {', '.join(VALID_ROLES)}"

    auth = _load_auth()
    users = auth.get("users", {})
    if username in users:
        return False, f"User '{username}' already exists"

    users[username] = {"password": _hash_password(password), "role": role}
    auth["users"] = users
    _save_auth(auth)
    logger.info(f"Auth: Created user '{username}' with role '{role}'")
    return True, f"User '{username}' created"


def delete_user(username: str) -> tuple[bool, str]:
    auth = _load_auth()
    users = auth.get("users", {})
    if username not in users:
        return False, "User not found"

    # Can't delete the last admin
    admin_count = sum(1 for u in users.values() if u.get("role") == "admin")
    if users[username].get("role") == "admin" and admin_count <= 1:
        return False, "Cannot delete the last admin user"

    del users[username]
    _save_auth(auth)
    logger.info(f"Auth: Deleted user '{username}'")
    return True, f"User '{username}' deleted"


def update_user_role(username: str, role: str) -> tuple[bool, str]:
    if role not in VALID_ROLES:
        return False, f"Invalid role. Must be one of: {', '.join(VALID_ROLES)}"

    auth = _load_auth()
    users = auth.get("users", {})
    user = users.get(username)
    if not user:
        return False, "User not found"

    old_role = user.get("role", "viewer")
    # Can't demote the last admin
    if old_role == "admin" and role != "admin":
        admin_count = sum(1 for u in users.values() if u.get("role") == "admin")
        if admin_count <= 1:
            return False, "Cannot demote the last admin user"

    user["role"] = role
    _save_auth(auth)
    logger.info(f"Auth: Updated '{username}' role to '{role}'")
    return True, f"Role updated to '{role}'"


def has_permission(user_role: str, required_role: str) -> bool:
    """Check if user_role meets the required_role level."""
    return ROLE_HIERARCHY.get(user_role, 0) >= ROLE_HIERARCHY.get(required_role, 99)


# --- Sessions (with role) ---

SESSION_TTL = 86400
_sessions: dict[str, tuple[str, str, float]] = {}  # token -> (username, role, created_at)


def generate_session_token() -> str:
    return secrets.token_hex(32)


def create_session(username: str) -> str:
    token = generate_session_token()
    role = get_user_role(username)
    _sessions[token] = (username, role, time.time())
    return token


def get_session_user(token: str) -> str | None:
    entry = _sessions.get(token)
    if entry is None:
        return None
    username, role, created_at = entry
    if time.time() - created_at > SESSION_TTL:
        del _sessions[token]
        return None
    return username


def get_session_role(token: str) -> str | None:
    entry = _sessions.get(token)
    if entry is None:
        return None
    username, role, created_at = entry
    if time.time() - created_at > SESSION_TTL:
        del _sessions[token]
        return None
    return role


def get_session_info(token: str) -> tuple[str | None, str | None]:
    """Get (username, role) from session token."""
    entry = _sessions.get(token)
    if entry is None:
        return None, None
    username, role, created_at = entry
    if time.time() - created_at > SESSION_TTL:
        del _sessions[token]
        return None, None
    return username, role


def delete_session(token: str) -> None:
    _sessions.pop(token, None)


def cleanup_expired_sessions() -> int:
    now = time.time()
    expired = [t for t, (_, _, created) in _sessions.items() if now - created > SESSION_TTL]
    for t in expired:
        del _sessions[t]
    return len(expired)
