"""
Bharat Online Cabs - Spatial Intelligence, Disambiguation & Fare Engine.

Engines:
  * Google Places API (New) - High precision real-time POI search across Nagpur
  * OpenStreetMap Nominatim - Autonomous local fallback when API key is unavailable
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

# Load .env
load_dotenv()
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()

# ----------------------------------------------------------------------------- config
DEFAULT_CITY = "Nagpur, Maharashtra"
DEFAULT_LAT, DEFAULT_LON = 21.1458, 79.0882

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "http://router.project-osrm.org/route/v1/driving"
GOOGLE_PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
USER_AGENT = "BharatOnlineCabs/1.0"
HTTP_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en"}

NOMINATIM_TIMEOUT_S = 6.0
OSRM_TIMEOUT_S = 5.0
GOOGLE_TIMEOUT_S = 7.0
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

NAGPUR_LOCALITIES = (
    "sitabuldi", "dharampeth", "dhantoli", "ramdaspeth", "civil lines", "sadar",
    "mahal", "gandhibagh", "nandanvan", "mominpura", "manish nagar", "besa",
    "hudkeshwar", "jaripatka", "kamptee", "itwari", "cotton market",
    "medical square", "seminary hills", "gayatri nagar", "pratap nagar",
    "wardha road", "trimurti nagar", "khamla", "bajaj nagar", "shankarnagar",
    "somalwada", "ayodhya nagar", "pardi", "wadi", "hingna", "lokmat square"
)


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


# ----------------------------------------------------------------------------- text helpers
def _normalize(name: str) -> str:
    s = name.lower().replace("'s", "s").replace("’s", "s")
    return re.sub(r"[^a-z0-9\u0900-\u097f]+", "", s)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _query_variants(query: str) -> list[str]:
    q = " ".join(query.split())
    variants = [q]
    spaced = re.sub(r"\b([A-Z])([A-Z][a-z])", r"\1 \2", q)
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", spaced)
    for v in (spaced, q.replace("-", " ")):
        if v not in variants:
            variants.append(v)
    return variants


def _extract_short_address(full_address: str) -> str:
    if not full_address:
        return "Nagpur"
    lower_addr = full_address.lower()
    for loc in NAGPUR_LOCALITIES:
        if loc in lower_addr:
            return loc.title()

    parts = [p.strip() for p in full_address.split(",") if p.strip()]
    cleaned = []
    for p in parts:
        if re.search(r"india|\d{6}|maharashtra", p, re.IGNORECASE):
            continue
        if p.lower() in ("nagpur", "nagpur urban taluka", "nagpur rural taluka"):
            continue
        cleaned.append(p)
    if not cleaned:
        return "Nagpur"
    return cleaned[-1]


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


def sanitize_search_query(raw_query: str) -> str:
    """Strips filler words and maps conversational phrasing to searchable POI terms."""
    q = raw_query.lower().strip()
    # Remove common speech filler phrases
    fillers = [
        "i would like to go to the",
        "i would like to go to",
        "i want to go to the",
        "i want to go to",
        "take me to the",
        "take me to",
        "shop near me",
        "nearest",
        "near me",
        "nearby",
        "any",
        "the",
        "please",
    ]
    for f in fillers:
        q = q.replace(f, "")
    q = q.strip()

    # Category synonym normalization
    category_map = {
        "jewelry": "jewellers",
        "jeweller": "jewellers",
        "jewellery": "jewellers",
        "jewelry farm": "jewellers",
        "jewelry shop": "jewellers",
        "jewellery shop": "jewellers",
        "jewelry store": "jewellers",
        "coffee shop": "cafe",
        "coffee": "cafe",
        "cafe": "cafe",
        "hospital": "hospital",
        "clinic": "hospital",
        "medical store": "pharmacy",
        "chemist": "pharmacy",
        "pharmacy": "pharmacy",
        "sweet shop": "sweets",
        "sweets": "sweets",
    }
    return category_map.get(q, q)


# ----------------------------------------------------------------------------- OSM Nominatim Fallback
def _nominatim(query: str, viewbox: str, limit: int = 20) -> list[dict]:
    global _last_nominatim_call
    with _rate_lock:
        wait = NOMINATIM_MIN_INTERVAL_S - (time.monotonic() - _last_nominatim_call)
        if wait > 0:
            time.sleep(wait)
        try:
            r = _session.get(
                NOMINATIM_URL,
                params={
                    "q": query, "format": "jsonv2", "limit": limit, "countrycodes": "in",
                    "viewbox": viewbox, "bounded": 1, "addressdetails": 1, "namedetails": 1,
                },
                timeout=NOMINATIM_TIMEOUT_S,
            )
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            return []
        finally:
            _last_nominatim_call = time.monotonic()


def _search_osm(query: str, user_lat: float, user_lon: float, radius_km: float = 15.0) -> list[dict]:
    vb = _viewbox(user_lat, user_lon, radius_km)
    candidates: list[dict] = []
    for variant in _query_variants(query):
        for item in _nominatim(variant, vb):
            try:
                lat, lon = float(item["lat"]), float(item["lon"])
            except (KeyError, ValueError):
                continue
            dist = calculate_distance(user_lat, user_lon, lat, lon)
            if dist > radius_km:
                continue
            name = _display_name(item)
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
    candidates.sort(key=lambda c: c["distance_km"])
    return candidates


def fallback_osm_search(query: str, user_lat: float, user_lon: float, radius_km: float = 15.0) -> list[dict]:
    """Autonomous fallback to local OSM engine if Google key fails or returns no results."""
    return _search_osm(query, user_lat, user_lon, radius_km=radius_km)


# ----------------------------------------------------------------------------- Google Places (New) Engine
def search_nearby_pois(raw_query: str, user_lat: float, user_lon: float,
                       radius_meters: float = 3000.0, radius_km: float | None = None) -> list[dict]:
    """
    Search POIs near user coordinates.
    Uses Google Places API (New) with circular bias and query normalization,
    falling back to OpenStreetMap Nominatim if API key is absent or unreachable.
    """
    if radius_km is not None:
        radius_meters = radius_km * 1000.0
    sanitized = sanitize_search_query(raw_query)
    if not sanitized:
        return []

    cache_key = (sanitized, round(user_lat, 3), round(user_lon, 3), round(radius_meters, -2))
    if cache_key in _search_cache:
        return [dict(c) for c in _search_cache[cache_key]]

    key = os.getenv("GOOGLE_PLACES_API_KEY")

    if key:
        url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": key,
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.rating"
        }
        payload = {
            "textQuery": sanitized,
            "locationBias": {
                "circle": {
                    "center": {"latitude": user_lat, "longitude": user_lon},
                    "radius": float(radius_meters)
                }
            }
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=5)
            data = resp.json()
            places = data.get("places", [])
            if places:
                results = []
                tokens = [w for w in re.findall(r'[a-zA-Z]{4,}', sanitized.lower())]
                is_cat = sanitized in ("jewellers", "cafe", "pharmacy", "sweets", "hospital")
                for p in places:
                    p_lat = p.get("location", {}).get("latitude")
                    p_lon = p.get("location", {}).get("longitude")
                    if p_lat is None or p_lon is None:
                        continue
                    name = p.get("displayName", {}).get("text", "Destination").strip()
                    addr = p.get("formattedAddress", "").strip()

                    # For specific named queries, ensure key distinctive tokens are matched
                    if not is_cat and tokens:
                        matched_tokens = sum(1 for t in tokens if t in name.lower() or t in addr.lower())
                        if matched_tokens < min(len(tokens), 2) and _similarity(sanitized, name) < 0.45:
                            continue
                        if "railway" in tokens and "railway" not in name.lower():
                            continue

                    dist = calculate_distance(user_lat, user_lon, p_lat, p_lon)
                    results.append({
                        "name": name,
                        "full_address": addr,
                        "short_address": _extract_short_address(addr),
                        "lat": round(p_lat, 7),
                        "lon": round(p_lon, 7),
                        "distance_km": round(dist, 2),
                        "category": "commercial/retail",
                        "rating": p.get("rating", 4.5),
                        "importance": 1.0,
                    })

                # Deduplicate identical pins within 200m with similar name
                deduped = []
                for r in results:
                    if not any(
                        calculate_distance(r["lat"], r["lon"], d["lat"], d["lon"]) < 0.2
                        and _similarity(r["name"], d["name"]) > 0.6
                        for d in deduped
                    ):
                        deduped.append(r)
                results = deduped

                results.sort(key=lambda x: x["distance_km"])
                print(f"[GOOGLE PLACES] Query '{sanitized}' returned {len(results)} places near ({user_lat}, {user_lon})")
                if results:
                    _search_cache[cache_key] = [dict(c) for c in results]
                    return results
        except Exception as e:
            print(f"[GOOGLE PLACES ERROR] {e}")

    # Fallback to local OSM engine if Google key fails
    osm_res = fallback_osm_search(sanitized, user_lat, user_lon, radius_km=max(5.0, radius_meters / 1000.0))
    _search_cache[cache_key] = [dict(c) for c in osm_res]
    return osm_res


# ----------------------------------------------------------------------------- Intent Evaluation
GENERIC_CATEGORY_KEYWORDS = (
    "shop", "store", "near me", "restaurant", "cafe", "coffee", "hotel", "hospital",
    "clinic", "atm", "any", "jewelry", "jeweller", "jewellery", "petrol pump", "dhaba",
    "bakery", "sweet", "bar", "pharmacy", "medical"
)


def evaluate_destination_intent(query: str, user_lat: float, user_lon: float,
                                radius_km: float = 15.0) -> dict:
    """
    Evaluates customer destination intent.
    Handles generic categories and chain outlets with enterprise concierge phrasing.
    """
    candidates = search_nearby_pois(query, user_lat, user_lon, radius_km)
    result = {
        "status": "NOT_FOUND",
        "query": query,
        "primary_match": None,
        "alternatives": [],
        "outlet_count": 0,
        "clarification_prompt": ""
    }

    if not candidates:
        result["clarification_prompt"] = (
            f"I couldn't find '{query}' within {radius_km:.0f} km in Nagpur. "
            "Could you tell me a nearby landmark or specific area name?"
        )
        return result

    nearest = candidates[0]
    rest = candidates[1:]
    result["outlet_count"] = len(candidates)
    result["primary_match"] = nearest
    result["alternatives"] = rest

    # Format human-friendly distance
    if nearest["distance_km"] < 1.0:
        dist_str = f"{int(nearest['distance_km'] * 1000)} meters"
    else:
        dist_str = f"{nearest['distance_km']:.1f} kilometers"

    where = f" in {nearest['short_address']}" if nearest.get("short_address") else ""

    sanitized = sanitize_search_query(query)
    is_generic = any(w in query.lower() for w in GENERIC_CATEGORY_KEYWORDS) or sanitized in (
        "jewellers", "cafe", "pharmacy", "sweets", "hospital"
    )

    max_spread = 0.0
    if len(candidates) > 1:
        max_spread = max(calculate_distance(candidates[0]["lat"], candidates[0]["lon"], c["lat"], c["lon"])
                         for c in candidates)

    if is_generic or (len(candidates) > 1 and max_spread > 0.5):
        result["status"] = "DISAMBIGUATION_NEEDED"
        cat_label = {
            "jewellers": "jeweler",
            "cafe": "cafe",
            "pharmacy": "pharmacy",
            "sweets": "sweet shop",
            "hospital": "hospital",
        }.get(sanitized, "option")

        where_text = f" in {nearest['short_address']}" if nearest.get("short_address") else ""
        if cat_label != "option":
            result["clarification_prompt"] = (
                f"The closest {cat_label} is {nearest['name']}{where_text}, about {dist_str} away. "
                f"Would you like to go there, or pick another from the map?"
            )
        else:
            result["clarification_prompt"] = (
                f"The closest option is {nearest['name']}{where_text}, about {dist_str} away. "
                f"Would you like to go there, or pick another from the map?"
            )
        return result

    # Single unambiguous landmark
    result["status"] = "EXACT"
    result["clarification_prompt"] = (
        f"Found {nearest['name']}{where}, about {dist_str} away. Shall I calculate the fare?"
    )
    return result


# ----------------------------------------------------------------------------- Routing & Fare
def calculate_fare(distance_km: float) -> float:
    """Bharat Online Cabs tariff: ₹50 base + ₹14/km, minimum ₹70."""
    return round(max(MIN_FARE_INR, BASE_FARE_INR + PER_KM_INR * distance_km), 2)


def calculate_trip_details(user_lat: float, user_lon: float, dest_lat: float, dest_lon: float,
                           destination_name: str) -> dict:
    """
    Road route via OSRM; falls back to Haversine * 1.3 on timeout/error.
    route_geometry is a GeoJSON LineString ([lon, lat] pairs) for the frontend map.
    """
    source = "osrm"
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
        distance_km = route["distance"] / 1000.0
        duration_mins = max(1, math.ceil(route["duration"] / 60.0))
        geometry = route["geometry"]
    except (requests.RequestException, ValueError, KeyError):
        source = "haversine_fallback"
        distance_km = calculate_distance(user_lat, user_lon, dest_lat, dest_lon) * ROAD_FACTOR
        duration_mins = max(1, math.ceil(distance_km / FALLBACK_SPEED_KMPH * 60))
        geometry = {
            "type": "LineString",
            "coordinates": [[user_lon, user_lat], [dest_lon, dest_lat]]
        }

    distance_km = round(distance_km, 2)
    return {
        "destination": destination_name,
        "distance_km": distance_km,
        "duration_mins": int(duration_mins),
        "fare_inr": calculate_fare(distance_km),
        "route_geometry": geometry,
        "route_source": source,
    }
