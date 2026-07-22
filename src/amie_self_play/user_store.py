from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
import unicodedata
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


PASSWORD_MIN_LENGTH = 10
PASSWORD_MAX_LENGTH = 128
USERNAME_MIN_LENGTH = 3
USERNAME_MAX_LENGTH = 32
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60


class UserStoreError(ValueError):
    """Base class for user-facing account errors."""


class UsernameTakenError(UserStoreError):
    """Raised when a normalized username is already registered."""


class InvalidCredentialsError(UserStoreError):
    """Raised when a username and password pair cannot be authenticated."""


@dataclass(frozen=True, slots=True)
class User:
    id: str
    username: str


def normalize_username(value: str) -> tuple[str, str]:
    username = unicodedata.normalize("NFKC", value).strip()
    if not USERNAME_MIN_LENGTH <= len(username) <= USERNAME_MAX_LENGTH:
        raise UserStoreError(
            f"用户名长度必须为 {USERNAME_MIN_LENGTH}-{USERNAME_MAX_LENGTH} 个字符"
        )
    if not all(character.isalnum() or character in "._-" for character in username):
        raise UserStoreError("用户名只能包含文字、数字、点、下划线或连字符")
    if not any(character.isalnum() for character in username):
        raise UserStoreError("用户名至少需要包含一个文字或数字")
    return username, username.casefold()


def validate_password(password: str) -> None:
    if not PASSWORD_MIN_LENGTH <= len(password) <= PASSWORD_MAX_LENGTH:
        raise UserStoreError(
            f"密码长度必须为 {PASSWORD_MIN_LENGTH}-{PASSWORD_MAX_LENGTH} 个字符"
        )


def _password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )


def _session_hash(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


class UserStore:
    def __init__(
        self, data_dir: Path, *, session_ttl_seconds: int = SESSION_TTL_SECONDS
    ) -> None:
        self.data_dir = data_dir
        self.database_path = data_dir / "users.sqlite3"
        self.prompts_dir = data_dir / "prompts"
        self.session_ttl_seconds = session_ttl_seconds

    def initialize(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.prompts_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.data_dir, 0o700)
        os.chmod(self.prompts_dir, 0o700)
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    username_key TEXT NOT NULL UNIQUE,
                    password_hash BLOB NOT NULL,
                    password_salt BLOB NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash BLOB PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS sessions_user_id
                    ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS sessions_expires_at
                    ON sessions(expires_at);
                """
            )
            connection.execute(
                "DELETE FROM sessions WHERE expires_at <= ?", (int(time.time()),)
            )
        os.chmod(self.database_path, 0o600)

    def register(self, username_value: str, password: str) -> User:
        username, username_key = normalize_username(username_value)
        validate_password(password)
        user = User(id=uuid.uuid4().hex, username=username)
        salt = os.urandom(16)
        password_hash = _password_hash(password, salt)
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO users (
                        id, username, username_key, password_hash,
                        password_salt, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user.id,
                        user.username,
                        username_key,
                        password_hash,
                        salt,
                        int(time.time()),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise UsernameTakenError("该用户名已被注册") from exc
        return user

    def authenticate(self, username_value: str, password: str) -> User:
        try:
            _, username_key = normalize_username(username_value)
        except UserStoreError:
            username_key = ""
        if len(password) > PASSWORD_MAX_LENGTH:
            password = ""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT id, username, password_hash, password_salt
                FROM users WHERE username_key = ?
                """,
                (username_key,),
            ).fetchone()
        salt = bytes(row["password_salt"]) if row is not None else bytes(16)
        supplied_hash = _password_hash(password, salt)
        expected_hash = (
            bytes(row["password_hash"]) if row is not None else bytes(32)
        )
        if row is None or not hmac.compare_digest(expected_hash, supplied_hash):
            raise InvalidCredentialsError("用户名或密码错误")
        return User(id=str(row["id"]), username=str(row["username"]))

    def create_session(self, user: User) -> str:
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            connection.execute(
                """
                INSERT INTO sessions (token_hash, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    _session_hash(token),
                    user.id,
                    now,
                    now + self.session_ttl_seconds,
                ),
            )
        return token

    def user_for_session(self, token: str | None) -> User | None:
        if not token:
            return None
        now = int(time.time())
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT users.id, users.username
                FROM sessions
                JOIN users ON users.id = sessions.user_id
                WHERE sessions.token_hash = ? AND sessions.expires_at > ?
                """,
                (_session_hash(token), now),
            ).fetchone()
            if row is None:
                connection.execute(
                    "DELETE FROM sessions WHERE token_hash = ?",
                    (_session_hash(token),),
                )
                return None
        return User(id=str(row["id"]), username=str(row["username"]))

    def delete_session(self, token: str | None) -> None:
        if not token:
            return
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "DELETE FROM sessions WHERE token_hash = ?", (_session_hash(token),)
            )

    def delete_user(self, user: User) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM users WHERE id = ?", (user.id,))

    def prompt_path(self, user: User) -> Path:
        return self.prompts_dir / user.id / "prompts.toml"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection
