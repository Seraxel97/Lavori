"""Tests per common.run_id."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

from common.run_id import _config_hash_short, _git_sha_short, generate

_TS = datetime(2026, 5, 1, 10, 8, 28, tzinfo=UTC)
_TS_STR = "20260501T100828"


class TestGenerate:
    def test_default_prefix_and_structure(self):
        rid = generate(ts=_TS)
        assert rid.startswith("run_")
        parts = rid.split("_")
        assert parts[1] == _TS_STR

    def test_custom_prefix(self):
        rid = generate(prefix="bench", ts=_TS)
        assert rid.startswith("bench_")

    def test_no_config_path_has_3_parts(self):
        rid = generate(ts=_TS)
        assert rid.count("_") == 2  # prefix, ts, sha

    def test_with_config_path_has_4_parts(self, tmp_path):
        cfg = tmp_path / "cfg.py"
        cfg.write_text("x = 1")
        rid = generate(config_path=cfg, ts=_TS)
        assert rid.count("_") == 3  # prefix, ts, sha, hash

    def test_nonexistent_config_path_omits_hash(self):
        rid = generate(config_path="/nonexistent/config.py", ts=_TS)
        assert rid.count("_") == 2

    def test_sha_length(self):
        with patch("common.run_id._git_sha_short", return_value="abcd1234"):
            rid = generate(ts=_TS, sha_len=8)
        parts = rid.split("_")
        assert parts[2] == "abcd1234"

    def test_config_hash_length(self, tmp_path):
        cfg = tmp_path / "cfg.py"
        cfg.write_text("x = 1")
        rid = generate(config_path=cfg, ts=_TS, hash_len=4)
        parts = rid.split("_")
        assert len(parts[3]) == 4

    def test_nogit_fallback(self, monkeypatch):
        monkeypatch.setattr("common.run_id._git_sha_short", lambda length=8: "nogit")
        rid = generate(ts=_TS)
        assert "nogit" in rid

    def test_deterministic_same_ts_same_config(self, tmp_path):
        cfg = tmp_path / "cfg.py"
        cfg.write_text("x = 1")
        r1 = generate(config_path=cfg, ts=_TS)
        r2 = generate(config_path=cfg, ts=_TS)
        assert r1 == r2

    def test_different_configs_different_hash(self, tmp_path):
        cfg1 = tmp_path / "cfg1.py"
        cfg1.write_text("x = 1")
        cfg2 = tmp_path / "cfg2.py"
        cfg2.write_text("x = 2")
        r1 = generate(config_path=cfg1, ts=_TS)
        r2 = generate(config_path=cfg2, ts=_TS)
        assert r1 != r2

    def test_ts_default_is_utc_now(self):
        before = datetime.now(UTC)
        rid = generate()
        after = datetime.now(UTC)
        ts_str = rid.split("_")[1]
        parsed = datetime.strptime(ts_str, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        assert before.replace(microsecond=0) <= parsed <= after.replace(microsecond=0)


class TestGitShortHelper:
    def test_returns_string(self):
        result = _git_sha_short()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_length_capped(self):
        result = _git_sha_short(length=4)
        assert len(result) <= 4

    def test_fallback_on_git_error(self, monkeypatch):
        import subprocess
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()))
        result = _git_sha_short()
        assert result == "nogit"


class TestConfigHashHelper:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("content")
        h1 = _config_hash_short(f, length=8)
        h2 = _config_hash_short(f, length=8)
        assert h1 == h2

    def test_length(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("content")
        assert len(_config_hash_short(f, length=6)) == 6

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.py"
        f1.write_text("aaa")
        f2 = tmp_path / "b.py"
        f2.write_text("bbb")
        assert _config_hash_short(f1) != _config_hash_short(f2)
