from __future__ import annotations

import stat

import pytest

from amie_self_play.user_store import (
    InvalidCredentialsError,
    UserStore,
    UserStoreError,
    UsernameTakenError,
)


def test_password_and_session_tokens_are_hashed_at_rest(tmp_path) -> None:
    store = UserStore(tmp_path / "accounts")
    store.initialize()
    user = store.register("测试用户", "private-password-123")
    token = store.create_session(user)

    database_bytes = store.database_path.read_bytes()
    assert b"private-password-123" not in database_bytes
    assert token.encode("ascii") not in database_bytes
    assert stat.S_IMODE(store.database_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.data_dir.stat().st_mode) == 0o700

    restarted_store = UserStore(store.data_dir)
    restarted_store.initialize()
    assert restarted_store.user_for_session(token) == user
    assert restarted_store.user_for_session("无效会话") is None
    restarted_store.delete_session(token)
    assert restarted_store.user_for_session(token) is None


def test_username_and_password_validation_and_casefolded_uniqueness(tmp_path) -> None:
    store = UserStore(tmp_path / "accounts")
    store.initialize()

    with pytest.raises(UserStoreError):
        store.register("ab", "valid-password-123")
    with pytest.raises(UserStoreError):
        store.register("valid-user", "short")

    store.register("Alice", "alice-password-123")
    with pytest.raises(UsernameTakenError):
        store.register("alice", "another-password-123")
    with pytest.raises(InvalidCredentialsError):
        store.authenticate("Alice", "wrong-password")


def test_each_user_has_a_distinct_prompt_path(tmp_path) -> None:
    store = UserStore(tmp_path / "accounts")
    store.initialize()
    first = store.register("first-user", "first-password-123")
    second = store.register("second-user", "second-password-123")

    assert store.prompt_path(first) != store.prompt_path(second)
    assert store.prompt_path(first).name == "prompts.toml"
    assert store.prompt_path(first).parent.name == first.id
