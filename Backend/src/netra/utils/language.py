"""Language detection helpers for NETRA (script-based, offline)."""

from __future__ import annotations

LANG_DETECT_RANGES: dict[str, list[tuple[str, str]]] = {
    "hi": [("\u0900", "\u097F")],  # Devanagari (hi/mr/ne/sa…)
    "bn": [("\u0980", "\u09FF")],
    "pa": [("\u0A00", "\u0A7F")],
    "gu": [("\u0A80", "\u0AFF")],
    "or": [("\u0B00", "\u0B7F")],
    "ta": [("\u0B80", "\u0BFF")],
    "te": [("\u0C00", "\u0C7F")],
    "kn": [("\u0C80", "\u0CFF")],
    "ml": [("\u0D00", "\u0D7F")],
    "ur": [("\u0600", "\u06FF")],
}


def detect_language(text: str) -> str:
    """Detect dominant Indic script language; default English for Latin text."""
    if not text or not text.strip():
        return "en"
    script_counts: dict[str, int] = {lang: 0 for lang in LANG_DETECT_RANGES}
    latin_count = 0
    total = 0
    for ch in text:
        if ch.isspace() or ch in ".,;:!?-()[]{}\"'0123456789":
            continue
        total += 1
        cp = ord(ch)
        if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            latin_count += 1
            continue
        for lang, ranges in LANG_DETECT_RANGES.items():
            for lo, hi in ranges:
                if ord(lo) <= cp <= ord(hi):
                    script_counts[lang] += 1
                    break
    if total == 0:
        return "en"
    best_lang = max(script_counts, key=script_counts.get)
    if script_counts[best_lang] > 0:
        return best_lang
    return "en"
