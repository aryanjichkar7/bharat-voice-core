"""
Bharat Online Cabs - Spatial Intelligence, Disambiguation & Fare Engine.

Engines:
  * OpenStreetMap Nominatim - Autonomous geocoding & POI search across Nagpur
  * OSRM Road Route & Dynamic Fare Engine - Turn-by-turn road polyline and BOC tariff
"""
from __future__ import annotations

import math
import os
import re
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger

from osm_indexer import get_index_stats, query_local_places

# Load .env
load_dotenv()

# ----------------------------------------------------------------------------- config
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Nagpur, Maharashtra")
DEFAULT_CITY_NAME = os.getenv("DEFAULT_CITY_NAME", DEFAULT_CITY.split(",")[0].strip())
DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", "21.1458"))
DEFAULT_LON = float(os.getenv("DEFAULT_LON", "79.0882"))

MAP_PROVIDER = os.getenv("MAP_PROVIDER", "osm")
OSM_GEOCODER_URL = os.getenv("OSM_GEOCODER_URL", "https://nominatim.openstreetmap.org/search").strip()
OSM_REVERSE_GEOCODER_URL = os.getenv("OSM_REVERSE_GEOCODER_URL", "https://nominatim.openstreetmap.org/reverse").strip()
OSM_ROUTING_URL = os.getenv("OSM_ROUTING_URL", "http://router.project-osrm.org/route/v1/driving").strip()

NOMINATIM_URL = OSM_GEOCODER_URL
NOMINATIM_REVERSE_URL = OSM_REVERSE_GEOCODER_URL
OSRM_URL = OSM_ROUTING_URL
USER_AGENT = "BharatOnlineCabs/1.0"
HTTP_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en"}

NOMINATIM_TIMEOUT_S = 3.0
OSRM_TIMEOUT_S = 4.0
NOMINATIM_MIN_INTERVAL_S = 1.05

BASE_FARE_INR = 50.0
PER_KM_INR = 14.0
MIN_FARE_INR = 70.0
ROAD_FACTOR = 1.3
FALLBACK_SPEED_KMPH = 25.0

EARTH_RADIUS_KM = 6371.0088
DEDUPE_RADIUS_KM = 0.05
CHAIN_SIMILARITY = 0.85
MAX_SPOKEN_OPTIONS = 3

_session = requests.Session()
_session.headers.update(HTTP_HEADERS)
_rate_lock = threading.Lock()
_last_nominatim_call = 0.0
_search_cache: dict[tuple, list[dict]] = {}

# ----------------------------------------------------------------------------- geometry
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _viewbox(lat: float, lon: float, radius_km: float) -> str:
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    return f"{lon - dlon},{lat + dlat},{lon + dlon},{lat - dlat}"


# ----------------------------------------------------------------------------- text helpers & similarity
def _normalize(name: str) -> str:
    s = name.lower().replace("'s", "s").replace("’s", "s")
    return re.sub(r"[^a-z0-9\u0900-\u097f]+", "", s)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _phonetic_skeleton(s: str) -> str:
    """Generates phonetic consonant skeleton for transliterated toponyms."""
    s = s.lower().strip()
    s = re.sub(r"c(?=[ei])", "s", s)
    s = re.sub(r"ph", "f", s)
    s = re.sub(r"w", "v", s)
    s = re.sub(r"(?:tion|sion|shun)\b", "shn", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    tokens = re.findall(r"[a-z0-9]+", s)
    res_tokens = []
    for tok in tokens:
        if not tok:
            continue
        first = tok[0]
        rest = re.sub(r"[aeiou]", "", tok[1:])
        res_tokens.append(first + rest)
    return " ".join(res_tokens)


def _token_jaccard(tokens1: list[str], tokens2: list[str]) -> float:
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


# ----------------------------------------------------------------------------- Configurable matching thresholds
EXACT_MATCH_NAME_THRESHOLD = 0.92
EXACT_MATCH_COVERAGE_THRESHOLD = 0.85
PLAUSIBLE_NAME_THRESHOLD = 0.65
PLAUSIBLE_COVERAGE_THRESHOLD = 0.50
POSSIBLE_MATCH_MIN_SCORE = 0.65
DISAMBIGUATION_MIN_SCORE = 0.60
DISAMBIGUATION_MAX_MARGIN = 0.12

GENERIC_STOPWORDS = {
    "the", "a", "an", "to", "for", "at", "in", "on", "of", "and", "&", "me", "us", "please"
}

GENERIC_TOPONYM_DESCRIPTORS = {
    "road", "rd", "street", "st", "marg", "chowk", "chauk", "square", "sq",
    "station", "stn", "stop", "nagar", "colony", "layout", "complex", "plaza",
    "point", "corner", "junction", "junc", "rly", "railway", "metro",
    "building", "tower", "towers", "center", "centre", "mall", "market",
    "bazaar", "bazar", "gate", "bridge", "pool", "garden", "gardens", "park",
    "house", "office", "hub", "lane", "circle"
}


def normalize_name_for_matching(text: str) -> str:
    """Standardizes text for matching without collapsing word boundaries."""
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r"['’]s\b", "", t)
    t = re.sub(r"[^\w\s\u0900-\u097f]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def token_match_score(q: str, c: str) -> float:
    """Computes calibrated similarity between an individual query token and candidate token."""
    if q == c:
        return 1.0
    l_max = max(len(q), len(c))
    l_min = min(len(q), len(c))
    ld = _levenshtein_distance(q, c)
    max_edits = 1 if l_max <= 5 else 2
    if ld <= max_edits:
        return max(0.0, 1.0 - (ld / l_max))
    if (q.startswith(c) or c.startswith(q)) and l_min >= 4:
        return l_min / l_max
    sq = _phonetic_skeleton(q)
    sc = _phonetic_skeleton(c)
    if len(sq) >= 3 and len(sc) >= 3 and _levenshtein_distance(sq, sc) <= 1:
        return 0.85
    return 0.0


def _compound_token_match(tokens: list[str], target: str) -> float:
    """Checks if a target string is a sequential compound of individual tokens (e.g. 'alpha' + 'place' -> 'alphaplace')."""
    if not tokens or not target:
        return 0.0
    pos = 0
    total_matched_len = 0
    matched_tokens = 0
    for q in tokens:
        sub_c = target[pos:]
        matched = False
        for l in range(max(1, len(q) - 1), min(len(sub_c) + 1, len(q) + 3)):
            cand_slice = sub_c[:l]
            if not cand_slice:
                continue
            if token_match_score(q, cand_slice) >= 0.75:
                pos += len(cand_slice)
                total_matched_len += len(cand_slice)
                matched_tokens += 1
                matched = True
                break
        if not matched:
            break
    if matched_tokens == len(tokens) and total_matched_len >= len(target) - 2:
        return total_matched_len / max(len(target), 1)
    return 0.0


def score_candidate(query: str, candidate: dict, user_lat: float, user_lon: float) -> dict:
    """
    Computes calibrated similarity score for a retrieved candidate place against the query.
    Signals combined:
    1. Token-level recall and coverage across candidate name and legitimate locality.
    2. Character edit distance (Levenshtein) and SequenceMatcher ratio on word-separated text.
    3. Generic phonetic consonant skeleton similarity.
    4. Legitimate locality / address keyword match (excluding candidate name tokens and city).
    5. Proximity score (ensuring geographic distance does NOT make unrelated names appear valid).
    """
    q_clean = query.lower().strip()
    c_name = candidate.get("name", "").strip()
    c_short = candidate.get("short_address", "").strip()
    dist_km = candidate.get("distance_km", 99.0)

    q_norm = normalize_name_for_matching(q_clean)
    c_norm = normalize_name_for_matching(c_name)

    # 1. Token extraction
    q_tokens = [t for t in q_norm.split() if t not in GENERIC_STOPWORDS and len(t) >= 2]
    c_tokens = [t for t in c_norm.split() if t not in GENERIC_STOPWORDS and len(t) >= 2]
    if not q_tokens:
        q_tokens = q_norm.split() or [q_clean]
    if not c_tokens:
        c_tokens = c_norm.split() or [c_name.lower()]

    # Legitimate locality tokens: tokens in short_address outside of candidate name and default city
    loc_norm = normalize_name_for_matching(c_short)
    c_name_tokens_set = set(c_norm.split())
    loc_tokens = [t for t in loc_norm.split() if t not in GENERIC_STOPWORDS and t not in c_name_tokens_set and t != DEFAULT_CITY_NAME.lower() and len(t) >= 3]

    # Substantive candidate tokens for precision calculation (excluding generic descriptors)
    subst_c_tokens = [c for c in c_tokens if c not in GENERIC_TOPONYM_DESCRIPTORS] or c_tokens

    # 2. Token-level matching matrix
    q_matches = []
    for q in q_tokens:
        best_t = max([token_match_score(q, c) for c in c_tokens] + [0.0])
        # Check locality if not matched in place name
        if best_t < 0.70 and loc_tokens and any(token_match_score(q, lt) >= 0.75 for lt in loc_tokens):
            best_t = max(best_t, 0.85)
        q_matches.append(best_t)
    token_coverage = sum(q_matches) / len(q_matches) if q_matches else 0.0

    c_matches = [max([token_match_score(q, c) for q in q_tokens] + [0.0]) for c in subst_c_tokens]
    token_precision = sum(c_matches) / len(c_matches) if c_matches else 0.0

    # Compound word segmentation check (e.g. multi-word split vs single compound)
    if len(c_tokens) == 1 and len(q_tokens) >= 2:
        comp_score = _compound_token_match(q_tokens, c_tokens[0])
        if comp_score >= 0.75:
            token_coverage = max(token_coverage, comp_score)
            token_precision = max(token_precision, comp_score)
    elif len(q_tokens) == 1 and len(c_tokens) >= 2:
        comp_score = _compound_token_match(c_tokens, q_tokens[0])
        if comp_score >= 0.75:
            token_coverage = max(token_coverage, comp_score)
            token_precision = max(token_precision, comp_score)

    # 3. Whole-phrase character and phonetic similarity
    lev_phrase = _levenshtein_distance(q_norm, c_norm)
    l_phrase_ratio = max(0.0, 1.0 - (lev_phrase / max(len(q_norm), len(c_norm), 1)))
    s_phrase_ratio = SequenceMatcher(None, q_norm, c_norm).ratio()
    char_sim = 0.5 * l_phrase_ratio + 0.5 * s_phrase_ratio

    q_skel = _phonetic_skeleton(q_norm)
    c_skel = _phonetic_skeleton(c_norm)
    phonetic_sim = SequenceMatcher(None, q_skel, c_skel).ratio() if (q_skel and c_skel) else 0.0

    # 4. Calibrated Name Similarity
    t_clean = re.sub(r"^the\s+", "", q_norm)
    c_clean = re.sub(r"^the\s+", "", c_norm)
    if q_norm == c_norm or (bool(t_clean) and t_clean == c_clean):
        name_sim = 1.0
    else:
        raw_name_sim = 0.40 * token_coverage + 0.20 * token_precision + 0.25 * char_sim + 0.15 * phonetic_sim
        if len(q_tokens) >= 2 and token_coverage < 0.50:
            name_sim = raw_name_sim * ((token_coverage / 0.50) ** 2)
        else:
            name_sim = raw_name_sim

    # 5. Locality match
    locality_sim = 1.0 if (loc_tokens and any(token_match_score(q, lt) >= 0.80 for q in q_tokens for lt in loc_tokens)) else 0.0

    # 6. Proximity score
    prox_score = max(0.0, 1.0 - (dist_km / 30.0))

    # Substantive query token match (ensures match is not solely on generic descriptors like 'road' or 'square')
    subst_q_tokens = [q for q in q_tokens if q not in GENERIC_TOPONYM_DESCRIPTORS]
    if subst_q_tokens:
        has_substantive_match = any(
            max([token_match_score(q, c) for c in c_tokens] + [0.0]) >= 0.70
            or (loc_tokens and any(token_match_score(q, lt) >= 0.75 for lt in loc_tokens))
            for q in subst_q_tokens
        ) or (len(c_tokens) == 1 and len(q_tokens) >= 2 and _compound_token_match(q_tokens, c_tokens[0]) >= 0.75) or (len(q_tokens) == 1 and len(c_tokens) >= 2 and _compound_token_match(c_tokens, q_tokens[0]) >= 0.75)
    else:
        has_substantive_match = (name_sim >= 0.85)

    single_token_plausible = True
    if len(q_tokens) == 1:
        is_cand_cat = any(cat in c_norm for cat in COMMON_CATEGORIES)
        if not is_cand_cat and name_sim < 0.65:
            single_token_plausible = False

    # 7. Defensible Plausibility Gate
    is_plausible = (
        name_sim >= PLAUSIBLE_NAME_THRESHOLD
        and has_substantive_match
        and single_token_plausible
        and (token_coverage >= PLAUSIBLE_COVERAGE_THRESHOLD or len(q_tokens) == 1)
    )

    if not is_plausible:
        # Distance must never make an unrelated name appear valid
        total_score = round(name_sim * 0.4, 4)
        explanation = f"Unrelated candidate: name_sim={name_sim:.2f}, token_coverage={token_coverage:.2f}"
    elif name_sim >= EXACT_MATCH_NAME_THRESHOLD and token_coverage >= EXACT_MATCH_COVERAGE_THRESHOLD:
        total_score = round(0.88 * name_sim + 0.04 * locality_sim + 0.08 * prox_score, 4)
        explanation = f"Strong match: name_sim={name_sim:.2f}, token_coverage={token_coverage:.2f}"
    else:
        total_score = round(0.75 * name_sim + 0.10 * locality_sim + 0.15 * prox_score, 4)
        explanation = f"Plausible match: name_sim={name_sim:.2f}, token_coverage={token_coverage:.2f}, dist={dist_km:.1f}km"

    return {
        "candidate": candidate,
        "name_sim": round(name_sim, 3),
        "token_coverage": round(token_coverage, 3),
        "char_sim": round(char_sim, 3),
        "phonetic_sim": round(phonetic_sim, 3),
        "locality_sim": round(locality_sim, 3),
        "prox_score": round(prox_score, 3),
        "is_plausible": is_plausible,
        "total_score": total_score,
        "explanation": explanation,
    }


def _query_variants(query: str) -> list[str]:
    q = " ".join(query.split())
    variants = [q]
    spaced = re.sub(r"\b([A-Z])([A-Z][a-z])", r"\1 \2", q)
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", spaced)
    for v in (spaced, q.replace("-", " "), q.replace(" ", "")):
        if v not in variants and len(v) >= 3:
            variants.append(v)
    q_low = q.lower()

    # 1. Soft initial 'c' before 'i' or 'e' to 's'
    if re.match(r"^c[ie]", q_low):
        variants.append("s" + q_low[1:])

    # 2. Double consonant collapse (tt->t, dd->d, ll->l, etc.)
    for v in list(variants):
        dedup = re.sub(r"([b-df-hj-np-tv-z])\1+", r"\1", v)
        if dedup not in variants:
            variants.append(dedup)

    # 3. Generic toponym transliteration alternations
    suffix_pairs = [
        (r"path\b", "peth"),
        (r"peth\b", "path"),
        (r"pet\b", "peth"),
        (r"peth\b", "pet"),
        (r"ch[ao]wk\b", "chauk"),
        (r"chauk\b", "chowk"),
        (r"rly\b", "railway"),
        (r"railway\b", "rly"),
        (r"pur\b", "pore"),
        (r"pore\b", "pur"),
    ]
    for v in list(variants):
        for pat, rep in suffix_pairs:
            cand = re.sub(pat, rep, v, flags=re.IGNORECASE)
            if cand != v and cand not in variants:
                variants.append(cand)

    # 4. Medial vowel shifts (i <-> a) when query contains generic toponym suffixes
    toponym_pattern = re.compile(r"(?:path|peth|pur|pore|chowk|chawk|chauk|wadi|vadi|nagar|ngr)\b", re.IGNORECASE)
    for v in list(variants):
        if toponym_pattern.search(v):
            for m in re.finditer(r"([b-df-hj-np-tv-z])i(?=([b-df-hj-np-tv-z]))", v, flags=re.IGNORECASE):
                start = m.start()
                alt = v[:start+1] + "a" + v[start+2:]
                if alt not in variants:
                    variants.append(alt)
                    for pat, rep in suffix_pairs:
                        cand = re.sub(pat, rep, alt, flags=re.IGNORECASE)
                        if cand != alt and cand not in variants:
                            variants.append(cand)

    return variants[:4]


def _extract_short_address(full_address: str) -> str:
    if not full_address:
        return DEFAULT_CITY_NAME
    parts = [p.strip() for p in full_address.split(",") if p.strip()]
    cleaned = []
    for p in parts:
        if re.search(r"india|\d{6}|maharashtra", p, re.IGNORECASE):
            continue
        if re.search(rf"^({re.escape(DEFAULT_CITY_NAME.lower())}|{re.escape(DEFAULT_CITY_NAME.lower())}\s+urban\s+taluka|{re.escape(DEFAULT_CITY_NAME.lower())}\s+rural\s+taluka|{re.escape(DEFAULT_CITY_NAME.lower())}\s+city)$", p, re.IGNORECASE):
            continue
        cleaned.append(p)
    if not cleaned:
        return DEFAULT_CITY_NAME
    if len(cleaned) == 1:
        return cleaned[0]
    return f"{cleaned[-2]}, {cleaned[-1]}" if len(cleaned) >= 2 else cleaned[0]


def _short_address(item: dict) -> str:
    addr = item.get("address", {})
    parts = []
    for key in ("road", "neighbourhood", "suburb", "city_district"):
        val = addr.get(key)
        if val and val not in parts:
            parts.append(val)
        if len(parts) == 2:
            break
    if not parts:
        parts = [p.strip() for p in item.get("display_name", "").split(",")[1:3]]
    return ", ".join(parts)


def _display_name(item: dict) -> str:
    nd = item.get("namedetails") or {}
    return nd.get("name:en") or item.get("name") or item.get("display_name", "").split(",")[0]


COMMON_CATEGORIES = {
    "restaurant", "cafe", "hospital", "clinic", "pharmacy", "bakery",
    "petrol pump", "atm", "bank", "hotel", "supermarket", "grocery",
    "gym", "mall", "cinema", "park", "salon", "dhaba", "bar", "pub",
    "jewellers", "sweets", "coffee", "store", "shop"
}

CORRECTION_PREAMBLES = [
    r"^(?:actually|change of plans|i changed my mind|no not that|not that|instead of that|instead|rather|on second thought|can we change destination to|change destination to|change to)[,\s]+",
    r"^(?:can you|could you|would you|please|kindly)\s+",
]

TRANSIT_INTENT_PATTERNS = [
    r"\b(?:can\s+(?:i|we|you)\s+)?(?:book|reserve|get|call|order|arrange)\s+(?:me\s+|us\s+|for\s+me\s+|for\s+us\s+)?(?:a\s+)?(?:cab|ride|taxi|car)?\s*(?:from\s+here\s+)?(?:to|towards|at|for|near)\s+(?:\b(?:the|a|an)\b\s+)?(.+)",
    r"\b(?:take|drive|drop|head|bring|go|reach|travel|heading|going)\s+(?:me\s+|us\s+)?(?:off\s+)?(?:from\s+here\s+)?(?:to|towards|at|for|near|into)\s+(?:\b(?:the|a|an)\b\s+)?(.+)",
    r"\b(?:how\s+much\s+(?:to\s+go\s+)?to|what\s+is\s+the\s+fare\s+to|what\'s\s+the\s+fare\s+to|fare\s+to)\s+(?:\b(?:the|a|an)\b\s+)?(.+)",
    r"\b(?:i\s+|we\s+)?(?:want|wanted|wants|wanna|need|needed|needs|wish|would\s+like|\'?d\s+like|plan|planning|gotta|gonna)\s+(?:to\s+reach\s+|(?:to\s+)?(?:go|travel|head|get|reach)\s+(?:to|towards|at|for|into)\s+|(?:to|towards|at|for)\s+)(?:\b(?:the|a|an)\b\s+)?(.+)",
    r"\b(?:can\s+you\s+)?(?:find|search(?:\s+for)?|look(?:\s+for)?|show)\s+(?:me\s+)?(?:\b(?:a|an|the|any|some)\b\s+)?(?:good|nearby|nearest|best)?\s*(.+?)(?:\s+(?:nearby|near\s+me|around\s+here|close\s+by))?$",
    r"^(?:to|towards|at|near|heading to|going to)\s+(?:\b(?:the|a|an)\b\s+)?(.+)",
    r"(?:kripya\s+|कृपया\s+)?(?:mujhe|muje|hume|humein|hamko|मुझे|मुजे|हमें|हमे|हमको)?\s*(.+?)\s*(?:jana hai|le chalo|pahuncha do|chhod do|chalo|जाना है|ले चलो|पहुंचा दो|छोड़ दो)$",
    r"(.+?)\s+(?:ke liye cab chahiye|cab chahiye|chahiye|चाहिए)$",
    r"(?:krupaya\s+|कृपया\s+)?(?:mala|aamhala|मला|आम्हाला)?\s*(.+?)\s*(?:la|s|कडे|ला)?\s*(?:jaycha ahe|jayche aahe|pohachva|neun soda|gheun chala|जायचे आहे|घेऊन चला|नेऊन सोडा)$",
]

DESTINATION_PATTERNS = TRANSIT_INTENT_PATTERNS  # Preserved alias for external consumers


def is_general_query(raw_text: str) -> bool:
    """Detects general inquiries (weather, time, assistant identity, greetings, gratitude, pauses) that are not cab booking requests."""
    t = raw_text.lower().strip().rstrip("?!., ")
    if not t:
        return False

    # If text carries explicit transit verbal frames or prepositions, it is a transit request, NOT general query
    if any(re.search(pat, t, re.IGNORECASE) for pat in TRANSIT_INTENT_PATTERNS):
        return False
    if re.search(r"\b(?:to|towards|reach|drop\s+(?:me\s+)?at|heading\s+for)\s+[a-zA-Z0-9\u0900-\u097f]{2,}", t):
        return False

    # 1. Informational & Capability queries
    if re.match(r"^(?:what|who|why|tell me|explain|how are you|how do you|how does|what can you|what are you)\b", t):
        if not re.match(r"^how\s+much\s+(?:to|for|is|does)\b", t):
            return True

    # 2. Greetings
    if re.match(r"^(?:hello|hi|hey|namaste|namaskar|good\s+morning|good\s+afternoon|good\s+evening|howdy)\b", t):
        return True

    # 3. Gratitude & Politeness
    if re.match(r"^(?:thank\s+you|thanks|thank\s+u|dhanyawad|dhanyavad|shukriya|much\s+appreciated)\b", t) or t in ("welcome", "you're welcome", "you are welcome"):
        return True

    # 4. Pacing & Pauses
    if re.match(r"^(?:wait|wait\s+a\s+(?:minute|second|sec|moment)|hold\s+on|just\s+a\s+(?:minute|second|sec|moment)|one\s+moment|give\s+me\s+a\s+(?:second|sec|minute|moment)|thoda\s+ruko|ek\s+minute|zara\s+thamba|thamba)\b", t):
        return True

    # 5. Audio, Repetition & Comprehension inquiries
    if re.match(r"^(?:can\s+you\s+hear\s+me|are\s+you\s+there|are\s+you\s+listening|can\s+you\s+repeat|repeat\s+that|what\s+did\s+you\s+say|pardon|pardon\s+me|awaz\s+nahi\s+aa\s+rahi|aiku\s+yet\s+nahi)\b", t):
        return True

    # 6. Indecision & Non-booking statements
    if t in ("i don't know", "i dont know", "dont know", "not sure", "nothing", "never mind", "kuch nahi", "kahich nahi", "nko"):
        return True

    return False


CARRIER_PATTERNS = [
    r"^(?:i\s+|we\s+)?(?:want|wanna|need|gotta|gonna|wish|would\s+like|\'?d\s+like|plan|planning)\s*(?:to\s+)?(?:go|travel|head|get|reach|take|book)?\s*(?:a\s+)?(?:cab|ride|taxi|car)?$",
    r"^(?:let\s*\'?s\s+|let\s+us\s+|can\s+(?:we|i)\s+|could\s+(?:we|i)\s+)?(?:go|head|travel|leave|start|move)$",
    r"^(?:take|drive|drop|head|bring|go|reach|travel)\s*(?:me|us)?(?:\s+off)?$",
    r"^(?:book|reserve|get|call|order|arrange)\s*(?:me\s+|us\s+|for\s+me\s+|for\s+us\s+)?(?:a\s+)?(?:cab|ride|taxi|car)?$",
    r"^(?:a\s+)?(?:cab|ride|taxi|car)$",
    r"^(?:go|travel|ride)$",
    r"^(?:mujhe|muje|hume|humein|hamko|मुझे|मुजे|हमें|हमे|हमको)?\s*(?:jana hai|le chalo|chalo|pahuncha do|chhod do|जाना है|ले चलो|चलो|पहुंचा दो|छोड़ दो)$",
    r"^(?:cab\s+chahiye|gaadi\s+chahiye|कैब\s+चाहिए|गाड़ी\s+चाहिए)$",
    r"^(?:mala|aamhala|मला|आम्हाला)?\s*(?:jaycha ahe|jayche aahe|gheun chala|neun soda|pohachva|जायचे आहे|घेऊन चला|नेऊन सोडा)$",
]

CARRIER_WORDS = {
    "go", "to go", "i wanna go", "wanna go", "want to go", "head", "travel",
    "leave", "reach", "get", "take me", "drop me", "cab", "ride", "taxi", "car",
    "a cab", "a ride", "a taxi", "me", "us", "here", "there", "somewhere",
    "anywhere", "now", "today", "please", "chalo", "jana hai", "le chalo",
    "jaycha ahe", "cab chahiye"
}


def is_transit_carrier_only(text: str) -> bool:
    """Checks whether the utterance consists entirely of transit verbs or carrier phrases without a destination entity."""
    t = text.lower().strip().rstrip(".?!, ")
    if not t:
        return True
    if t in CARRIER_WORDS:
        return True
    return any(re.match(pat, t, re.IGNORECASE) for pat in CARRIER_PATTERNS)


def extract_destination_entity(raw_query: str) -> str:
    """
    Extracts destination entity from free-form natural language across English, Hindi, and Marathi.
    Handles varied sentence structures, bare destinations, conversational preambles, and corrections.
    Returns empty string if query is a general query, carrier-only phrase, or contains no destination entity.
    """
    if not raw_query:
        return ""
    if is_general_query(raw_query) or is_transit_carrier_only(raw_query):
        return ""
    t = raw_query.strip().rstrip(".?!, ")

    # 1. Strip conversational & corrective preambles
    for pat in CORRECTION_PREAMBLES:
        t = re.sub(pat, "", t, flags=re.IGNORECASE).strip()

    if is_transit_carrier_only(t):
        return ""

    # 2. Match generalized transit verbal frames
    for pat in TRANSIT_INTENT_PATTERNS:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            cand = re.sub(r"\s+(?:please|pls|fast|now|kripya|jaldi|lavkar|right\s+now|asap)$", "", cand, flags=re.IGNORECASE)
            cand = re.sub(r"\s+by\s+\d+(?::\d+)?(?:\s*[ap]m)?$", "", cand, flags=re.IGNORECASE)
            if cand.lower().endswith("la") and len(cand) > 4:
                cand = cand[:-2].strip()
            elif cand.endswith("ला") and len(cand) > 2:
                cand = cand[:-2].strip()
            if is_transit_carrier_only(cand):
                return ""
            return cand

    # 3. Prepositional fallback heuristic
    fallback_m = re.search(r"\b(?:to|towards|reach|drop\s+(?:me\s+)?at|heading\s+for|for)\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z0-9\s\u0900-\u097f]{2,})$", t, re.IGNORECASE)
    if fallback_m:
        cand = fallback_m.group(1).strip()
        cand = re.sub(r"\s+(?:please|pls|fast|now|kripya|jaldi|lavkar|right\s+now|asap)$", "", cand, flags=re.IGNORECASE).strip()
        if len(cand) >= 2 and not any(cand.lower() == w for w in ("here", "there", "it", "now", "today", "please", "me", "us")):
            if is_transit_carrier_only(cand):
                return ""
            return cand

    # 4. Bare destination / unadorned place name
    cand = re.sub(r"^(?:the|a|an)\s+", "", t, flags=re.IGNORECASE).strip()
    cand = re.sub(r"\s+(?:please|pls|fast|now|kripya|jaldi|lavkar|right\s+now|asap)$", "", cand, flags=re.IGNORECASE).strip()
    if is_transit_carrier_only(cand):
        return ""
    return cand


def parse_natural_intent_and_destination(raw_text: str) -> dict:
    """
    Parses natural language requests into structured intent, destination entity, and category.
    Classifies cancellation, confirmation, general non-booking queries, and ride booking intent.
    """
    if not raw_text or not raw_text.strip():
        return {"intent": "UNKNOWN", "destination": None, "category": None, "raw": ""}

    t = raw_text.strip().rstrip(".?!, ")
    t_lower = t.lower()

    # 1. Cancellation intent
    if any(p in t_lower for p in ("cancel", "radd", "band karo", "stop", "dont book", "don't book", "never mind", "nahi chahiye", "nako")):
        return {"intent": "CANCELLATION", "destination": None, "category": None, "raw": t}

    # 2. General non-booking queries
    if is_general_query(t):
        return {"intent": "GENERAL_QUERY", "destination": None, "category": None, "raw": t}

    # 3. Confirmation intent
    words = set(re.findall(r"\b[a-zA-Z']+\b", t_lower))
    confirm_words = {
        "yes", "confirm", "sure", "yep", "yeah", "haan", "ha", "ho", "done",
        "correct", "right", "proceed", "ok", "okay", "alright", "perfect",
        "fine", "great", "chalel", "kardo"
    }
    confirm_phrases = (
        "book it", "book this", "please book", "book cab", "book a cab", "book the cab",
        "confirm booking", "confirm ride", "kar do", "theek hai", "chalo", "sahi hai",
        "that's right", "thats right", "go ahead", "sounds good", "looks good",
        "that works", "works for me", "let's do it", "lets do it", "let's go", "lets go",
        "please do", "go for it", "sure thing", "all right", "ha chalel", "ho chalel"
    )
    if (words & confirm_words) or any(p in t_lower for p in confirm_phrases):
        if not re.search(r"\b(?:to|towards|reach|drop\s+(?:me\s+)?at|heading\s+for)\s+[a-zA-Z0-9\u0900-\u097f]{2,}", t_lower):
            return {"intent": "CONFIRMATION", "destination": None, "category": None, "raw": t}

    # 4. Extract destination entity
    extracted = extract_destination_entity(t)
    cleaned_dest = re.sub(r"\s+(?:please|pls|fast|now|kripya|jaldi|lavkar|right\s+now)$", "", extracted, flags=re.IGNORECASE).strip()

    # Reject conversational fillers from becoming destinations
    if cleaned_dest.lower() in GENERIC_STOPWORDS or cleaned_dest.lower() in ("here", "there", "it", "now", "today", "please", "me", "us", "cab", "ride", "taxi", "car", "something", "anything"):
        return {"intent": "UNKNOWN", "destination": None, "category": None, "raw": t}

    is_cat = cleaned_dest.lower() in COMMON_CATEGORIES or any(c in cleaned_dest.lower() for c in COMMON_CATEGORIES)
    has_booking_verb = bool(re.search(r"\b(?:book|reserve|get|call|arrange|order)\b", t_lower))

    intent = "BOOK_CAB" if has_booking_verb else "DESTINATION_SEARCH"
    return {
        "intent": intent if len(cleaned_dest) >= 2 else "UNKNOWN",
        "destination": cleaned_dest if len(cleaned_dest) >= 2 else None,
        "category": cleaned_dest if is_cat else None,
        "raw": t
    }


def sanitize_search_query(raw_query: str) -> str:
    """Strips conversational wrapper phrasing, articles, and maps synonyms to POI terms."""
    extracted = extract_destination_entity(raw_query)
    q = extracted.lower().strip()
    # Strip leading generic articles/prepositions/qualifiers
    q = re.sub(r"^(?:the|a|an|to|at|for)\s+", "", q)
    q = re.sub(r"^(?:good|best|nearby|nearest|any|some)\s+", "", q)
    # Strip common filler suffixes using parameterized default city name
    q = re.sub(rf"\s+(?:near\s+me|nearby|around\s+here|close\s+by|please|in\s+{re.escape(DEFAULT_CITY_NAME.lower())})$", "", q)
    q = q.strip()

    # Category synonym & plural normalization
    category_map = {
        # Dining & Food
        "restaurants": "restaurant",
        "restaurant": "restaurant",
        "coffee shop": "cafe",
        "coffee shops": "cafe",
        "coffee": "cafe",
        "cafes": "cafe",
        "cafe": "cafe",
        "bakeries": "bakery",
        "bakery": "bakery",
        "sweet shop": "sweets",
        "sweet shops": "sweets",
        "sweets": "sweets",
        "dhaba": "dhaba",
        "dhabas": "dhaba",
        "bar": "bar",
        "bars": "bar",
        "pub": "pub",
        "pubs": "pub",

        # Medical
        "hospitals": "hospital",
        "hospital": "hospital",
        "clinics": "hospital",
        "clinic": "hospital",
        "pharmacies": "pharmacy",
        "pharmacy": "pharmacy",
        "medical store": "pharmacy",
        "medical stores": "pharmacy",
        "chemist": "pharmacy",
        "chemists": "pharmacy",
        "chemist shop": "pharmacy",

        # Fuel & Financial
        "petrol pump": "petrol pump",
        "petrol pumps": "petrol pump",
        "gas station": "petrol pump",
        "gas stations": "petrol pump",
        "fuel pump": "petrol pump",
        "fuel station": "petrol pump",
        "atm": "atm",
        "atms": "atm",
        "bank": "bank",
        "banks": "bank",

        # Retail & Shopping
        "hotels": "hotel",
        "hotel": "hotel",
        "supermarket": "supermarket",
        "supermarkets": "supermarket",
        "grocery": "grocery",
        "groceries": "grocery",
        "grocery store": "grocery",
        "grocery stores": "grocery",
        "gym": "gym",
        "gyms": "gym",
        "fitness center": "gym",
        "mall": "mall",
        "malls": "mall",
        "shopping mall": "mall",
        "cinema": "cinema",
        "cinemas": "cinema",
        "movie theater": "cinema",
        "movie theatre": "cinema",
        "theatre": "cinema",
        "theater": "cinema",
        "park": "park",
        "parks": "park",
        "salon": "salon",
        "salons": "salon",
        "parlour": "salon",
        "parlor": "salon",
        "jewelry": "jewellers",
        "jeweller": "jewellers",
        "jewellers": "jewellers",
        "jewellery": "jewellers",
        "jewelry farm": "jewellers",
        "jewelry shop": "jewellers",
        "jewellery shop": "jewellers",
        "jewelry store": "jewellers",
    }
    return category_map.get(q, q)


# ----------------------------------------------------------------------------- OSM Nominatim Fallback
def _nominatim(query: str, viewbox: str, limit: int = 20, timeout: float = NOMINATIM_TIMEOUT_S) -> list[dict]:
    global _last_nominatim_call, _last_search_meta
    with _rate_lock:
        wait = NOMINATIM_MIN_INTERVAL_S - (time.monotonic() - _last_nominatim_call)
        if wait > 0:
            time.sleep(wait)
        t_req_start = time.monotonic()
        try:
            r = _session.get(
                NOMINATIM_URL,
                params={
                    "q": query, "format": "jsonv2", "limit": limit, "countrycodes": "in",
                    "viewbox": viewbox, "bounded": 1, "addressdetails": 1, "namedetails": 1,
                },
                timeout=timeout,
            )
            elapsed = time.monotonic() - t_req_start
            _last_search_meta["provider_status"] = r.status_code
            _last_search_meta["last_request_elapsed_s"] = elapsed
            if r.status_code == 200:
                _last_search_meta["provider_error"] = None
                items = r.json()
                logger.info(f"[OSM] Query {query!r} -> HTTP {r.status_code} in {elapsed:.2f}s ({len(items)} items)")
                return items
            else:
                _last_search_meta["provider_error"] = f"Nominatim returned HTTP {r.status_code}"
                logger.warning(f"[OSM] Query {query!r} -> HTTP {r.status_code} in {elapsed:.2f}s")
                return []
        except requests.exceptions.Timeout as e:
            elapsed = time.monotonic() - t_req_start
            _last_search_meta["provider_status"] = 504
            _last_search_meta["provider_error"] = f"Nominatim timeout after {elapsed:.2f}s"
            _last_search_meta["last_request_elapsed_s"] = elapsed
            logger.warning(f"[OSM] Query {query!r} -> Timeout after {elapsed:.2f}s: {e}")
            return []
        except requests.exceptions.RequestException as e:
            elapsed = time.monotonic() - t_req_start
            status = getattr(getattr(e, "response", None), "status_code", 503)
            _last_search_meta["provider_status"] = status
            _last_search_meta["provider_error"] = f"Nominatim network error: {e}"
            _last_search_meta["last_request_elapsed_s"] = elapsed
            logger.warning(f"[OSM] Query {query!r} -> RequestException ({status}) after {elapsed:.2f}s: {e}")
            return []
        except ValueError as e:
            elapsed = time.monotonic() - t_req_start
            _last_search_meta["provider_status"] = 500
            _last_search_meta["provider_error"] = f"Invalid JSON response: {e}"
            _last_search_meta["last_request_elapsed_s"] = elapsed
            return []
        finally:
            _last_nominatim_call = time.monotonic()


def reverse_geocode_coordinate(lat: float, lon: float) -> dict:
    """Reverse-geocodes a GPS pickup coordinate using OpenStreetMap reverse geocoder."""
    global _last_nominatim_call
    with _rate_lock:
        wait = NOMINATIM_MIN_INTERVAL_S - (time.monotonic() - _last_nominatim_call)
        if wait > 0:
            time.sleep(wait)
        try:
            r = _session.get(
                NOMINATIM_REVERSE_URL,
                params={
                    "lat": lat,
                    "lon": lon,
                    "format": "jsonv2",
                    "addressdetails": 1,
                },
                timeout=NOMINATIM_TIMEOUT_S,
            )
            r.raise_for_status()
            data = r.json()
            return {
                "display_name": data.get("display_name", ""),
                "short_address": _short_address(data),
                "lat": float(data.get("lat", lat)),
                "lon": float(data.get("lon", lon)),
                "address": data.get("address", {}),
            }
        except Exception as e:
            return {
                "display_name": f"{lat:.4f}° N, {lon:.4f}° E, {DEFAULT_CITY_NAME}",
                "short_address": f"{DEFAULT_CITY_NAME} Mobility Core",
                "lat": lat,
                "lon": lon,
                "address": {},
                "error": str(e),
            }
        finally:
            _last_nominatim_call = time.monotonic()


def _search_osm(query: str, user_lat: float, user_lon: float, radius_km: float = 15.0, is_category: bool = False, max_variants: int = 2) -> list[dict]:
    t_search_start = time.monotonic()
    search_budget_s = 4.0
    vb = _viewbox(user_lat, user_lon, radius_km)
    candidates: list[dict] = []
    max_candidates = 10 if is_category else 4

    variants = [query] if is_category else _query_variants(query)[:max_variants]
    for variant in variants:
        elapsed = time.monotonic() - t_search_start
        remaining = search_budget_s - elapsed
        if remaining < 0.6:
            logger.info(f"[OSM] Search budget reached (remaining={remaining:.2f}s), skipping variant {variant!r}")
            break

        req_timeout = min(NOMINATIM_TIMEOUT_S, remaining)
        items = _nominatim(variant, vb, limit=20, timeout=req_timeout)

        # If provider returned an error or timeout, do NOT attempt sequential variants
        if _last_search_meta.get("provider_error") or _last_search_meta.get("provider_status") in (408, 429, 500, 502, 503, 504):
            logger.warning(f"[OSM] Provider error {_last_search_meta.get('provider_status')}, breaking search loop")
            break

        for item in items:
            try:
                lat, lon = float(item["lat"]), float(item["lon"])
            except (KeyError, ValueError):
                continue
            dist = calculate_distance(user_lat, user_lon, lat, lon)
            if dist > radius_km:
                continue
            name = _display_name(item)
            is_road = item.get("type") in ("trunk", "primary", "secondary", "tertiary", "residential", "motorway")
            if is_road and not is_category and any(_normalize(c["name"]) == _normalize(name) for c in candidates):
                continue
            if any(_normalize(c["name"]) == _normalize(name)
                   and calculate_distance(lat, lon, c["lat"], c["lon"]) < DEDUPE_RADIUS_KM
                   for c in candidates):
                continue
            candidates.append({
                "name": name,
                "full_address": item.get("display_name", ""),
                "short_address": _short_address(item),
                "lat": round(lat, 7),
                "lon": round(lon, 7),
                "distance_km": round(dist, 2),
                "category": f"{item.get('category', '')}/{item.get('type', '')}",
                "rating": 4.5,
                "importance": float(item.get("importance") or 0.0),
            })
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= (5 if is_category else 2):
            break

    candidates.sort(key=lambda c: c["distance_km"])
    return candidates


def fallback_osm_search(query: str, user_lat: float, user_lon: float, radius_km: float = 15.0, is_category: bool = False) -> list[dict]:
    """OSM geocoding search within local radius, bounding sequential variants to stay within processing budget."""
    return _search_osm(query, user_lat, user_lon, radius_km=radius_km, is_category=is_category, max_variants=2)


_last_search_meta: dict = {
    "provider_used": "Local OpenStreetMap SQLite Engine",
    "provider_status": 200,
    "provider_error": None,
    "raw_places_count": 0,
    "filtered_places_count": 0,
}


# ----------------------------------------------------------------------------- OpenStreetMap Dynamic POI Engine
def search_nearby_pois(raw_query: str, user_lat: float, user_lon: float,
                       radius_meters: float = 3000.0, radius_km: float | None = None) -> list[dict]:
    """
    Search POIs near user coordinates using Local OpenStreetMap SQLite Engine.
    Handles both named place resolution and business-category discovery with zero external latency.
    Falls back gracefully if the local database is missing or unpopulated.
    """
    global _last_search_meta
    if radius_km is not None:
        radius_meters = radius_km * 1000.0
    effective_radius_km = max(5.0, radius_meters / 1000.0)

    sanitized = sanitize_search_query(raw_query)
    if not sanitized:
        return []

    cache_key = (sanitized, round(user_lat, 3), round(user_lon, 3), round(radius_meters, -2))
    if cache_key in _search_cache:
        return [dict(c) for c in _search_cache[cache_key]]

    is_cat = sanitized in COMMON_CATEGORIES or any(cat in sanitized for cat in COMMON_CATEGORIES)

    # 1. Primary backend: Local OpenStreetMap SQLite Engine
    db_stats = get_index_stats()
    if db_stats.get("database_exists") and db_stats.get("total_rows", 0) > 0:
        t0 = time.monotonic()
        try:
            results = query_local_places(sanitized, user_lat, user_lon, radius_km=effective_radius_km)
            elapsed = time.monotonic() - t0
            _last_search_meta["provider_used"] = "Local OpenStreetMap SQLite Engine"
            _last_search_meta["provider_status"] = 200
            _last_search_meta["provider_error"] = None
            _last_search_meta["raw_places_count"] = len(results)
            _last_search_meta["filtered_places_count"] = len(results)
            _last_search_meta["last_request_elapsed_s"] = elapsed
            logger.info(f"[Local OSM] Query {sanitized!r} -> {len(results)} items in {elapsed*1000:.1f}ms")
            if results:
                _search_cache[cache_key] = [dict(c) for c in results]
            return results
        except Exception as e:
            logger.error(f"[Local OSM] Query exception: {e}")
            _last_search_meta["provider_error"] = str(e)
            _last_search_meta["provider_status"] = 500
            return []

    # 2. Local database is not initialized
    # Constraint: Do not claim local search works against an empty database.
    allow_remote = os.getenv("ENABLE_PUBLIC_OSM_FALLBACK", "false").lower() in ("true", "1")
    if not allow_remote:
        logger.warning("[OSM] Local place database not initialized or empty.")
        _last_search_meta["provider_used"] = "Local OpenStreetMap SQLite Engine"
        _last_search_meta["provider_status"] = 503
        _last_search_meta["provider_error"] = "Local place database not initialized. Run python import_osm_places.py to build index."
        _last_search_meta["raw_places_count"] = 0
        _last_search_meta["filtered_places_count"] = 0
        return []

    # 3. Optional fallback to public Nominatim only if explicitly permitted
    _last_search_meta = {
        "provider_used": "OpenStreetMap Nominatim Engine",
        "provider_status": 200,
        "provider_error": None,
        "raw_places_count": 0,
        "filtered_places_count": 0,
    }
    try:
        results = fallback_osm_search(sanitized, user_lat, user_lon, radius_km=effective_radius_km, is_category=is_cat)
        _last_search_meta["raw_places_count"] = len(results)
        _last_search_meta["filtered_places_count"] = len(results)
        if _last_search_meta.get("provider_status") == 200 and not _last_search_meta.get("provider_error"):
            _search_cache[cache_key] = [dict(c) for c in results]
        return results
    except Exception as e:
        _last_search_meta["provider_error"] = str(e)
        _last_search_meta["provider_status"] = 500
        return []


# ----------------------------------------------------------------------------- Intent Evaluation
GENERIC_CATEGORY_KEYWORDS = (
    "shop", "store", "near me", "restaurant", "cafe", "coffee", "hotel", "hospital",
    "clinic", "atm", "bank", "any", "jewelry", "jeweller", "jewellery", "petrol pump",
    "gas station", "dhaba", "bakery", "sweet", "bar", "pub", "pharmacy", "medical",
    "supermarket", "grocery", "gym", "mall", "cinema", "park", "salon"
)


def evaluate_destination_intent(query: str, user_lat: float, user_lon: float,
                                radius_km: float = 15.0,
                                candidates: list[dict] | None = None) -> dict:
    """
    Evaluates customer destination intent using candidate-grounded typo and phonetic recovery.
    Distinguishes 5 outcomes: EXACT_MATCH, POSSIBLE_MATCH, MULTIPLE_CANDIDATES, NO_RESULTS, PROVIDER_ERROR.
    """
    clean_dest = extract_destination_entity(query)
    if not clean_dest:
        return {
            "status": "NOT_FOUND",
            "resolution_type": "NO_RESULTS",
            "query": query,
            "primary_match": None,
            "suggested_match": None,
            "alternatives": [],
            "outlet_count": 0,
            "clarification_prompt": f"Where in {DEFAULT_CITY_NAME} would you like to go?",
            "provider_status": 200,
            "provider_error": None,
            "confidence_score": 0.0,
        }
    if candidates is None:
        candidates = search_nearby_pois(clean_dest, user_lat, user_lon, radius_km=radius_km)
    meta = dict(_last_search_meta)

    result = {
        "status": "NOT_FOUND",
        "resolution_type": "NO_RESULTS",
        "query": clean_dest or query,
        "primary_match": None,
        "suggested_match": None,
        "alternatives": [],
        "outlet_count": 0,
        "clarification_prompt": "",
        "provider_status": meta.get("provider_status", 200),
        "provider_error": meta.get("provider_error"),
        "confidence_score": 0.0,
    }

    display_q = clean_dest if clean_dest else query

    if not candidates:
        if meta.get("provider_status") in (408, 429, 403, 500, 502, 503, 504) or meta.get("provider_error"):
            result["status"] = "PROVIDER_ERROR"
            result["resolution_type"] = "PROVIDER_ERROR"
            result["provider_status"] = meta.get("provider_status", 504)
            result["provider_error"] = meta.get("provider_error") or "Map search provider timeout"
            result["clarification_prompt"] = (
                "Our map search service is temporarily unavailable. "
                f"Could you specify a nearby main road or landmark in {DEFAULT_CITY_NAME} so I can route you?"
            )
        else:
            result["status"] = "NOT_FOUND"
            result["resolution_type"] = "NO_RESULTS"
            result["provider_status"] = 200
            result["clarification_prompt"] = (
                f"I couldn't find '{display_q}' within {radius_km:.0f} km. "
                "Could you tell me a nearby landmark or major road?"
            )
        return result

    # Score all candidates against user query
    scored = [score_candidate(clean_dest, c, user_lat, user_lon) for c in candidates]
    scored.sort(key=lambda s: s["total_score"], reverse=True)
    plausible_scored = [s for s in scored if s["is_plausible"]]
    ranked_candidates = [s["candidate"] for s in scored]

    top = scored[0]
    top_cand = top["candidate"]
    top_score = top["total_score"]
    second_score = scored[1]["total_score"] if len(scored) > 1 else 0.0

    result["outlet_count"] = len(ranked_candidates)
    result["confidence_score"] = top_score

    # Human-friendly distance
    if top_cand["distance_km"] < 1.0:
        dist_str = f"{int(top_cand['distance_km'] * 1000)} meters"
    else:
        dist_str = f"{top_cand['distance_km']:.1f} kilometers"
    where = f" in {top_cand['short_address']}" if top_cand.get("short_address") else ""

    sanitized = sanitize_search_query(query)
    q_lower = query.lower()
    is_generic = (
        sanitized in COMMON_CATEGORIES
        or any(re.search(r"\b" + re.escape(cat) + r"\b", q_lower) for cat in COMMON_CATEGORIES)
        or any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in GENERIC_CATEGORY_KEYWORDS)
    )

    is_commercial = any(cat in top_cand.get("category", "") for cat in ("shop", "amenity", "commercial", "tourism/hotel"))
    is_chain = (
        is_commercial
        and len(candidates) > 1
        and sum(1 for c in candidates if _similarity(c["name"], top_cand["name"]) >= 0.85) >= 2
    )

    if is_generic or (is_chain and top_score < 0.95):
        result["status"] = "DISAMBIGUATION_NEEDED"
        result["resolution_type"] = "MULTIPLE_CANDIDATES"
        result["primary_match"] = top_cand
        result["alternatives"] = ranked_candidates[1:]
        cat_label_map = {
            "jewellers": "jeweler",
            "cafe": "cafe",
            "pharmacy": "pharmacy",
            "sweets": "sweet shop",
            "hospital": "hospital",
            "restaurant": "restaurant",
            "bakery": "bakery",
            "petrol pump": "petrol pump",
            "atm": "ATM",
            "bank": "bank",
            "hotel": "hotel",
            "supermarket": "supermarket",
            "grocery": "grocery store",
            "gym": "gym",
            "mall": "shopping mall",
            "cinema": "cinema",
            "park": "park",
            "salon": "salon",
            "dhaba": "dhaba",
            "bar": "bar",
            "pub": "pub",
        }
        cat_label = cat_label_map.get(sanitized, sanitized if (sanitized in COMMON_CATEGORIES and not is_chain) else "option")
        result["clarification_prompt"] = (
            f"The closest {cat_label} is {top_cand['name']}{where}, about {dist_str} away. "
            f"Would you like to go there, or pick another from the map?"
        )
        return result

    # Outcome: No defensible candidate meets the acceptance threshold -> NO_RESULTS
    if not plausible_scored or top_score < POSSIBLE_MATCH_MIN_SCORE:
        result["status"] = "NOT_FOUND"
        result["resolution_type"] = "NO_RESULTS"
        result["primary_match"] = None
        result["suggested_match"] = None
        result["alternatives"] = []
        result["confidence_score"] = top_score
        result["clarification_prompt"] = (
            f"I couldn't find a destination matching '{display_q}' within {radius_km:.0f} km. "
            "Could you tell me a nearby landmark or major road?"
        )
        return result

    # From here on, top candidate is defensibly grounded and plausible
    result["primary_match"] = top_cand
    result["alternatives"] = [s["candidate"] for s in plausible_scored[1:]]

    # Outcome: EXACT_MATCH (strictly requires exact normalized match without typo/phonetic drift)
    target_norm = normalize_name_for_matching(clean_dest)
    top_name_norm = normalize_name_for_matching(top_cand["name"])
    t_clean = re.sub(r"^the\s+", "", target_norm)
    c_clean = re.sub(r"^the\s+", "", top_name_norm)
    is_exact_name = (target_norm == top_name_norm or (bool(t_clean) and t_clean == c_clean))
    if is_exact_name and not (len(plausible_scored) > 1 and top_score - second_score < 0.08 and second_score >= 0.80):
        result["status"] = "EXACT"
        result["resolution_type"] = "EXACT_MATCH"
        result["clarification_prompt"] = (
            f"Found {top_cand['name']}{where}, about {dist_str} away. Shall I calculate the fare?"
        )
        return result

    # Outcome: Multiple genuinely plausible close candidates -> MULTIPLE_CANDIDATES
    if len(plausible_scored) > 1 and (top_score - second_score < DISAMBIGUATION_MAX_MARGIN) and (second_score >= DISAMBIGUATION_MIN_SCORE):
        result["status"] = "DISAMBIGUATION_NEEDED"
        result["resolution_type"] = "MULTIPLE_CANDIDATES"
        opts_summary = []
        for idx, s in enumerate(plausible_scored[:3], 1):
            c = s["candidate"]
            c_loc = f" in {c['short_address']}" if c.get("short_address") else ""
            opts_summary.append(f"{idx}. {c['name']}{c_loc} ({c['distance_km']:.1f} km)")
        options_text = ", ".join(opts_summary)
        result["clarification_prompt"] = (
            f"I found multiple locations for '{display_q}': {options_text}. Which one would you prefer?"
        )
        return result

    # Outcome: Plausible single typo / phonetic match -> POSSIBLE_MATCH
    result["status"] = "POSSIBLE_MATCH"
    result["resolution_type"] = "POSSIBLE_MATCH"
    result["suggested_match"] = top_cand
    result["clarification_prompt"] = f"Did you mean {top_cand['name']}{where}?"
    return result


# ----------------------------------------------------------------------------- Routing & Fare
def calculate_fare(distance_km: float) -> float:
    """Bharat Online Cabs tariff: ₹50 base + ₹14/km, minimum ₹70."""
    return round(max(MIN_FARE_INR, BASE_FARE_INR + PER_KM_INR * distance_km), 2)


def calculate_trip_details(user_lat: float, user_lon: float, dest_lat: float, dest_lon: float,
                           destination_name: str) -> dict:
    """
    Road route via OSRM (OSM_ROUTING_URL).
    route_geometry is a GeoJSON LineString ([lon, lat] pairs) for the frontend map.
    Returns honest road route, distance, duration, and fare.
    If routing fails or is unreachable, returns a recoverable error without fabricating distance.
    """
    try:
        r = _session.get(
            f"{OSRM_URL}/{user_lon},{user_lat};{dest_lon},{dest_lat}",
            params={"overview": "full", "geometries": "geojson", "alternatives": "false"},
            timeout=OSRM_TIMEOUT_S,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            raise ValueError(f"OSRM code={data.get('code')}")
        route = data["routes"][0]
        distance_km = round(route["distance"] / 1000.0, 2)
        duration_mins = max(1, math.ceil(route["duration"] / 60.0))
        geometry = route["geometry"]
        return {
            "status": "OK",
            "destination": destination_name,
            "dest": {"lat": dest_lat, "lon": dest_lon},
            "distance_km": distance_km,
            "duration_mins": int(duration_mins),
            "fare_inr": calculate_fare(distance_km),
            "route_geometry": geometry,
            "route_source": "osrm",
        }
    except (requests.RequestException, ValueError, KeyError) as e:
        return {
            "status": "ERROR",
            "error_code": "ROUTING_UNAVAILABLE",
            "error_message": f"Unable to calculate driving route ({e}).",
            "destination": destination_name,
            "dest": {"lat": dest_lat, "lon": dest_lon},
            "distance_km": None,
            "duration_mins": None,
            "fare_inr": None,
            "route_geometry": None,
            "route_source": "failed",
        }
