"""Offline unit tests — no live ML server required (CI-safe)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from netra.utils.language import detect_language
from netra.utils.location import (
    chunk_matches_location,
    get_state_from_pincode,
    normalize_geo,
)
from netra.utils.readiness import build_readiness, check_faiss_index


class TestLanguageDetect:
    def test_english(self):
        assert detect_language("What is PM-KISAN scheme?") == "en"

    def test_hindi_devanagari(self):
        assert detect_language("पीएम किसान योजना क्या है") == "hi"

    def test_tamil(self):
        assert detect_language("வணக்கம்") == "ta"

    def test_empty_defaults_en(self):
        assert detect_language("") == "en"
        assert detect_language("   ") == "en"


class TestPincodeState:
    def test_rajasthan(self):
        name, code = get_state_from_pincode("302001")
        assert name == "Rajasthan"
        assert code == "RJ"

    def test_tamil_nadu(self):
        name, code = get_state_from_pincode("600001")
        assert name == "Tamil Nadu"

    def test_invalid_length(self):
        assert get_state_from_pincode("123") == (None, None)

    def test_non_digit(self):
        assert get_state_from_pincode("abcdef") == (None, None)

    def test_goa(self):
        name, _ = get_state_from_pincode("403001")
        assert name == "Goa"

    def test_gujarat_36_37(self):
        assert get_state_from_pincode("360001")[0] == "Gujarat"
        assert get_state_from_pincode("370001")[0] == "Gujarat"


class TestGeoFilter:
    def test_national_chunk_always_matches(self):
        assert chunk_matches_location({"state": "", "district": ""}, "Rajasthan", "Jaipur")

    def test_state_mismatch_rejected(self):
        assert not chunk_matches_location(
            {"state": "Bihar", "district": ""},
            "Rajasthan",
            "Jaipur",
        )

    def test_state_match_accepted(self):
        assert chunk_matches_location(
            {"state": "Rajasthan", "district": "Jaipur"},
            "Rajasthan",
            "",
        )

    def test_district_mismatch_rejected(self):
        assert not chunk_matches_location(
            {"state": "Rajasthan", "district": "Udaipur"},
            "Rajasthan",
            "Jaipur",
        )

    def test_normalize_geo(self):
        assert normalize_geo("  Rajasthan ") == "rajasthan"


class TestReadiness:
    def test_faiss_missing(self, tmp_path: Path):
        report = check_faiss_index(tmp_path)
        assert report["ok"] is False
        assert report["chunks"] == 0

    def test_faiss_present(self, tmp_path: Path):
        (tmp_path / "index.faiss").write_bytes(b"fake")
        (tmp_path / "metadata.json").write_text(json.dumps([{"doc_id": "a"}, {"doc_id": "b"}]))
        report = check_faiss_index(tmp_path)
        assert report["ok"] is True
        assert report["chunks"] == 2

    def test_build_readiness_pipeline_required(self, tmp_path: Path):
        report = build_readiness(
            pipeline_ready=False,
            faiss_dir=tmp_path,
            require_ollama=False,
            require_faiss=False,
        )
        assert report["ready"] is False
        assert report["status"] == "not_ready"

    def test_build_readiness_pipeline_ok_without_ollama(self, tmp_path: Path):
        report = build_readiness(
            pipeline_ready=True,
            faiss_dir=tmp_path,
            require_ollama=False,
            require_faiss=False,
        )
        assert report["ready"] is True
        assert report["checks"]["pipeline"]["ok"] is True
