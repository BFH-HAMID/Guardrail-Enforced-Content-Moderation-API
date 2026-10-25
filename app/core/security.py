"""Secret handling: API key hashing, user-id pseudonymisation, at-rest encryption.

Design notes
------------
* API keys are never stored in plaintext. We store ``sha256(pepper || key)`` and
  compare digests with a constant-time compare, so a DB dump is not a credential
  dump (this mirrors how password and token hashing is done in production).
* A short ``key_id`` prefix is stored alongside the digest so an operator can find
  and revoke a key without being able to recover it.
* Optional at-rest encryption of the moderated text uses Fernet; we fail loudly if
  the caller turns encryption on without a key instead of silently storing plaintext.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import string
from typing import Protocol

from app.core.config import get_settings

KEY_PREFIX = "gmk_"  # guardrail moderation key
_ID_ALPHABET = string.ascii_letters + string.digits


def generate_api_key() -> str:
    """Create a new API key. Shown to the operator exactly once."""
    body = "".join(secrets.choice(_ID_ALPHABET) for _ in range(32))
    return f"{KEY_PREFIX}{body}"


def key_id_of(key: str) -> str:
    """Stable, non-secret identifier (first 12 chars) used for lookup and revocation."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _pepper() -> bytes:
    return get_settings().api_key_pepper.get_secret_value().encode("utf-8")


def hash_api_key(key: str) -> str:
    """``sha256(pepper || key)`` - the only representation we persist."""
    return hashlib.sha256(_pepper() + key.encode("utf-8")).hexdigest()


def verify_api_key(key: str, expected_hash: str) -> bool:
    try:
        return hmac.compare_digest(hash_api_key(key), expected_hash)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False


def hash_user_id(user_id: str) -> str:
    """Pseudonymise ``metadata.user_id`` so the log DB is not a user profile."""
    salt = get_settings().user_id_hash_salt.get_secret_value().encode("utf-8")
    return hmac.new(salt, user_id.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def hash_text(text: str) -> str:
    """Content fingerprint: lets us detect repeats without storing the text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class TextCipher(Protocol):
    def encrypt(self, plaintext: str) -> str: ...

    def decrypt(self, ciphertext: str) -> str: ...


class FernetCipher:
    """``cryptography``-backed cipher; imported lazily so it stays optional."""

    scheme = "fernet"

    def __init__(self, key: str | None = None) -> None:
        raw_key = key or get_settings().encryption_key.get_secret_value()
        if not raw_key:
            raise RuntimeError(
                "GUARD_ENCRYPT_STORED_TEXT=true requires GUARD_ENCRYPTION_KEY "
                "(generate one with: python -c 'from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())')"
            )
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError("encrypted storage needs the 'crypto' extra: pip install cryptography") from exc
        self._fernet = Fernet(raw_key.encode("utf-8") if isinstance(raw_key, str) else raw_key)

    def encrypt(self, plaintext: str) -> str:
        if plaintext == "":
            return plaintext
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        if not ciphertext or not ciphertext.startswith("gAAAAA"):
            return ciphertext  # already plaintext (written before encryption was enabled)
        return self._fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8", "replace")


class IdentityCipher:
    """No-op cipher used when ``GUARD_ENCRYPT_STORED_TEXT=false``."""

    scheme = "none"

    def encrypt(self, plaintext: str) -> str:
        return plaintext

    def decrypt(self, ciphertext: str) -> str:
        return ciphertext


def get_cipher() -> TextCipher:
    settings = get_settings()
    if settings.encrypt_stored_text:
        return FernetCipher()
    return IdentityCipher()


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")
