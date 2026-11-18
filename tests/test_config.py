import pytest
import yaml
from pydantic import ValidationError

from erp_duplicate_resolver.config import (
    DEFAULT_SETTINGS,
    _expand_env,
    load_abbreviations,
    load_settings,
)


def test_default_settings_load(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "anthropic:some-model")
    settings = load_settings()
    assert settings.thresholds.low < settings.thresholds.high
    assert settings.llm.model == "anthropic:some-model"
    assert settings.llm.usable


def test_missing_llm_model_disables_llm_instead_of_failing(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    settings = load_settings()
    assert settings.llm.model is None
    assert not settings.llm.usable


def test_env_expansion_with_default_and_embedding(monkeypatch):
    monkeypatch.setenv("FOO", "bar")
    monkeypatch.delenv("UNSET_VAR", raising=False)
    assert _expand_env("${FOO}") == "bar"
    assert _expand_env("${UNSET_VAR:-fallback}") == "fallback"
    assert _expand_env("${UNSET_VAR}") is None
    assert _expand_env("x-${FOO}-${UNSET_VAR}") == "x-bar-"
    assert _expand_env({"a": ["${FOO}", 1]}) == {"a": ["bar", 1]}


def _write_settings(tmp_path, **patch):
    raw = yaml.safe_load(DEFAULT_SETTINGS.open(encoding="utf-8"))
    for section, values in patch.items():
        raw[section].update(values)
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_weights_must_sum_to_one(tmp_path):
    with pytest.raises(ValidationError):
        load_settings(_write_settings(tmp_path, scoring={"w_string": 0.9}))


def test_low_threshold_must_not_exceed_high(tmp_path):
    with pytest.raises(ValidationError):
        load_settings(_write_settings(tmp_path, thresholds={"low": 0.95}))


def test_unknown_blocking_strategy_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load_settings(_write_settings(tmp_path, blocking={"strategies": ["soundex"]}))


def test_abbreviations_cover_spec_examples():
    abbr = load_abbreviations()
    assert abbr["verz."] == "verzinkt"
    assert abbr["schr."] == "schraube"
    assert abbr["edelst."] == "edelstahl"
