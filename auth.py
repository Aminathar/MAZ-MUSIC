"""
MAZ — Authentication utilities
JWT token creation/verification and bcrypt password hashing.

Requires:
    pip install python-jose[cryptography] bcrypt
"""

import os
from datetime import datetime, timedelta
from typing import Optional

ALGORITHM  = "HS256"
TOKEN_DAYS = 7

_env_key = os.environ.get("MAZ_SECRET_KEY", "")
if _env_key:
    SECRET_KEY = _env_key
else:
    import secrets as _secrets
    import logging as _logging
    SECRET_KEY = _secrets.token_hex(32)
    _logging.getLogger("maz.auth").warning(
        "MAZ_SECRET_KEY is not set — generated a random JWT secret for this session. "
        "All tokens will be invalidated on restart. "
        "Set MAZ_SECRET_KEY in your .env file to persist sessions."
    )

try:
    import bcrypt as _bcrypt
    from jose import JWTError, jwt as _jwt
    _available = True
except ImportError:
    _available = False
    _bcrypt = None


def _require():
    if not _available:
        raise RuntimeError(
            "Auth libraries missing. Run: pip install python-jose[cryptography] bcrypt"
        )


def hash_password(plain: str) -> str:
    """Return bcrypt hash of plain-text password."""
    _require()
    return _bcrypt.hashpw(plain.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the stored bcrypt hash."""
    _require()
    return _bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_token(user_id: int, role: str) -> str:
    """Return a signed JWT valid for TOKEN_DAYS days."""
    _require()
    exp = datetime.utcnow() + timedelta(days=TOKEN_DAYS)
    return _jwt.encode(
        {"sub": str(user_id), "role": role, "exp": exp},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def decode_token(token: str) -> Optional[dict]:
    """Return decoded payload dict, or None if invalid/expired."""
    _require()
    try:
        return _jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:
        return None
