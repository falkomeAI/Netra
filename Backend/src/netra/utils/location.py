"""PIN / state / district helpers for location-scoped RAG."""

from __future__ import annotations

# First 1–2 digits of Indian PIN → (state_name, state_code)
PINCODE_STATE_MAP: dict[str, tuple[str, str]] = {
    "1": ("Delhi", "DL"), "2": ("Delhi", "DL"),
    "11": ("Delhi", "DL"), "12": ("Haryana", "HR"), "13": ("Haryana", "HR"),
    "14": ("Punjab", "PB"), "15": ("Punjab", "PB"), "16": ("Punjab", "PB"),
    "17": ("Himachal Pradesh", "HP"),
    "18": ("Jammu & Kashmir", "JK"), "19": ("Jammu & Kashmir", "JK"),
    "20": ("Uttar Pradesh", "UP"), "21": ("Uttar Pradesh", "UP"),
    "22": ("Uttar Pradesh", "UP"), "23": ("Uttar Pradesh", "UP"),
    "24": ("Uttar Pradesh", "UP"), "25": ("Uttar Pradesh", "UP"),
    "26": ("Uttar Pradesh", "UP"), "27": ("Uttar Pradesh", "UP"),
    "28": ("Uttar Pradesh", "UP"),
    "30": ("Rajasthan", "RJ"), "31": ("Rajasthan", "RJ"),
    "32": ("Rajasthan", "RJ"), "33": ("Rajasthan", "RJ"),
    "34": ("Rajasthan", "RJ"),
    # Gujarat PIN ranges 36xxxx–39xxxx (not Haryana/AP)
    "36": ("Gujarat", "GJ"), "37": ("Gujarat", "GJ"),
    "38": ("Gujarat", "GJ"), "39": ("Gujarat", "GJ"),
    "40": ("Maharashtra", "MH"), "41": ("Maharashtra", "MH"),
    "42": ("Maharashtra", "MH"), "43": ("Maharashtra", "MH"),
    "44": ("Maharashtra", "MH"),
    "45": ("Madhya Pradesh", "MP"), "46": ("Madhya Pradesh", "MP"),
    "47": ("Madhya Pradesh", "MP"), "48": ("Madhya Pradesh", "MP"),
    "49": ("Chhattisgarh", "CG"),
    "50": ("Telangana", "TS"), "51": ("Telangana", "TS"),
    "52": ("Andhra Pradesh", "AP"), "53": ("Andhra Pradesh", "AP"),
    "56": ("Karnataka", "KA"), "57": ("Karnataka", "KA"),
    "58": ("Karnataka", "KA"), "59": ("Karnataka", "KA"),
    "60": ("Tamil Nadu", "TN"), "61": ("Tamil Nadu", "TN"),
    "62": ("Tamil Nadu", "TN"), "63": ("Tamil Nadu", "TN"),
    "64": ("Tamil Nadu", "TN"),
    "67": ("Kerala", "KL"), "68": ("Kerala", "KL"), "69": ("Kerala", "KL"),
    "70": ("West Bengal", "WB"), "71": ("West Bengal", "WB"),
    "72": ("West Bengal", "WB"), "73": ("West Bengal", "WB"),
    "74": ("West Bengal", "WB"),
    "75": ("Odisha", "OR"), "76": ("Odisha", "OR"), "77": ("Odisha", "OR"),
    "78": ("Assam", "AS"), "79": ("Northeast", "NE"),
    "80": ("Bihar", "BR"), "81": ("Bihar", "BR"),
    "82": ("Bihar", "BR"), "83": ("Bihar", "BR"), "84": ("Bihar", "BR"),
    "85": ("Jharkhand", "JH"),
    # Expanded coverage
    "10": ("Delhi", "DL"),
    "35": ("Andaman & Nicobar", "AN"),
    "54": ("Andhra Pradesh", "AP"), "55": ("Andhra Pradesh", "AP"),
    "65": ("Puducherry", "PY"), "66": ("Puducherry", "PY"),
    "737": ("Sikkim", "SK"),  # rare 3-digit prefix usage via loop below
    "744": ("Andaman & Nicobar", "AN"),
    "682": ("Lakshadweep", "LD"),
    "160": ("Chandigarh", "CH"),
    "194": ("Ladakh", "LA"),
    "248": ("Uttarakhand", "UK"), "249": ("Uttarakhand", "UK"),
    "263": ("Uttarakhand", "UK"),
    "403": ("Goa", "GA"),
}


def get_state_from_pincode(pincode: str) -> tuple[str | None, str | None]:
    """Resolve state from a 6-digit PIN code using prefix matching."""
    if len(pincode) != 6 or not pincode.isdigit():
        return None, None
    for prefix_len in (3, 2, 1):
        prefix = pincode[:prefix_len]
        if prefix in PINCODE_STATE_MAP:
            return PINCODE_STATE_MAP[prefix]
    return None, None


def normalize_geo(value: str | None) -> str:
    """Normalize state/district labels for metadata matching."""
    if not value:
        return ""
    return " ".join(str(value).strip().lower().split())


def chunk_matches_location(
    meta: dict,
    state: str | None = None,
    district: str | None = None,
) -> bool:
    """
    Return True if a KB chunk is allowed for the user's location.

    Rules:
    - National / untagged chunks (empty state) always match.
    - If state is set on the chunk, it must match the user state.
    - If both sides have district, districts must match; otherwise state match is enough.
    """
    want_state = normalize_geo(state)
    want_district = normalize_geo(district)
    chunk_state = normalize_geo(meta.get("state"))
    chunk_district = normalize_geo(meta.get("district"))

    # Untagged / national documents are always eligible
    if not chunk_state and not chunk_district:
        return True

    if want_state:
        if chunk_state and chunk_state != want_state:
            return False
    if want_district and chunk_district:
        if chunk_district != want_district:
            return False
    return True
