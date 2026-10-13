"""Phase 2 test suite for spatial_engine.py using live Nagpur data (Nominatim + OSRM)."""
import math
import sys
import time
from unittest import mock

import requests

import spatial_engine as se

# Reference points (OSM coordinates)
SITABULDI = (21.1402262, 79.0871588)     # Sitabuldi station
CIVIL_LINES = (21.1549282, 79.0789320)   # Civil Lines Road
CITY_CENTER = (se.DEFAULT_LAT, se.DEFAULT_LON)

results: list[tuple[str, bool, str]] = []


def banner(title: str) -> None:
    print(f"\n{'═' * 78}\n  {title}\n{'═' * 78}")


def check(label: str, cond: bool, detail: str = "") -> None:
    results.append((label, bool(cond), detail))
    print(f"   {'✅' if cond else '❌'} {label}" + (f"  [{detail}]" if detail else ""))


def show_candidates(cands, limit=6) -> None:
    print(f"   {'#':<3}{'Name':<26}{'Area':<38}{'Dist':>8}")
    for i, c in enumerate(cands[:limit], 1):
        print(f"   {i:<3}{c['name'][:25]:<26}{c['short_address'][:37]:<38}{c['distance_km']:>6.2f}km")
    if len(cands) > limit:
        print(f"   ... +{len(cands) - limit} more")


def run(name, fn) -> None:
    t0 = time.time()
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        check(f"{name} raised no exception", False, f"{type(e).__name__}: {e}")
    print(f"   ⏱  {time.time() - t0:.1f}s")


# ----------------------------------------------------------------------------- tests
def test_haversine():
    banner("TEST 0 · Haversine sanity")
    d = se.calculate_distance(*SITABULDI, *SITABULDI)
    check("Zero distance for identical points", d == 0.0, f"{d}")
    d = se.calculate_distance(0, 0, 0, 1)
    check("1° of longitude at equator ≈ 111.2 km", abs(d - 111.195) < 0.05, f"{d:.3f} km")
    d = se.calculate_distance(*SITABULDI, *CIVIL_LINES)
    check("Sitabuldi→Civil Lines straight line 1–3 km", 1 < d < 3, f"{d:.2f} km")


def test_chain():
    banner("TEST 1 · Ambiguous chain: 'Haldiram' and 'DMart' from city centre")
    for query in ("Haldiram", "DMart"):
        res = se.evaluate_destination_intent(query, *CITY_CENTER)
        print(f"\n   Query: '{query}' → status={res['status']}  outlets={res['outlet_count']}")
        show_candidates([res["primary_match"]] + res["alternatives"] if res["primary_match"] else [])
        print(f"   🗣  {res['clarification_prompt']}")
        dists = [res["primary_match"]["distance_km"]] + [a["distance_km"] for a in res["alternatives"]]
        check(f"{query}: DISAMBIGUATION_NEEDED", res["status"] == "DISAMBIGUATION_NEEDED")
        check(f"{query}: ≥2 outlets found", res["outlet_count"] >= 2, f"{res['outlet_count']}")
        check(f"{query}: nearest outlet first", dists == sorted(dists), f"nearest={dists[0]} km")
        check(f"{query}: prompt mentions nearest", res["primary_match"]["name"] in res["clarification_prompt"])


def test_landmark():
    banner("TEST 2 · Specific landmark: 'Nagpur Railway Station'")
    res = se.evaluate_destination_intent("Nagpur Railway Station", *SITABULDI)
    pm = res["primary_match"] or {}
    print(f"   status={res['status']}  match={pm.get('name')} @ ({pm.get('lat')}, {pm.get('lon')})")
    print(f"   address: {pm.get('full_address', '')[:90]}")
    print(f"   🗣  {res['clarification_prompt']}")
    check("Status EXACT", res["status"] == "EXACT")
    check("English name resolved", "Railway Station" in pm.get("name", ""), pm.get("name", ""))
    check("Within 2 km of Sitabuldi", pm.get("distance_km", 99) < 2, f"{pm.get('distance_km')} km")

    res = se.evaluate_destination_intent("Xyzzy Nonexistent Plaza", *SITABULDI)
    check("Gibberish query → NOT_FOUND", res["status"] == "NOT_FOUND", res["clarification_prompt"][:50])


def _verify_trip(trip, label):
    coords = trip["route_geometry"]["coordinates"]
    expected = round(max(se.MIN_FARE_INR, se.BASE_FARE_INR + se.PER_KM_INR * trip["distance_km"]), 2)
    print(f"   Destination : {trip['destination']}")
    print(f"   Source      : {trip['route_source']}")
    print(f"   Distance    : {trip['distance_km']} km")
    print(f"   ETA         : {trip['duration_mins']} min")
    print(f"   Fare        : ₹{trip['fare_inr']:.2f}  (₹{se.BASE_FARE_INR:.0f} + ₹{se.PER_KM_INR:.0f} × "
          f"{trip['distance_km']} km, min ₹{se.MIN_FARE_INR:.0f})")
    print(f"   Route pts   : {len(coords)}  first={coords[0]}  last={coords[-1]}")
    check(f"{label}: distance 1–6 km", 1 < trip["distance_km"] < 6, f"{trip['distance_km']} km")
    check(f"{label}: ETA 1–30 min", 1 <= trip["duration_mins"] <= 30, f"{trip['duration_mins']} min")
    check(f"{label}: fare math correct", math.isclose(trip["fare_inr"], expected), f"₹{expected}")
    check(f"{label}: GeoJSON LineString [lon,lat]",
          trip["route_geometry"]["type"] == "LineString" and len(coords) >= 2
          and abs(coords[0][0] - SITABULDI[1]) < 0.01 and abs(coords[-1][1] - CIVIL_LINES[0]) < 0.01)


def test_trip():
    banner("TEST 3 · Trip: Sitabuldi → Civil Lines (OSRM)")
    trip = se.calculate_trip_details(*SITABULDI, *CIVIL_LINES, "Civil Lines")
    _verify_trip(trip, "OSRM")
    check("OSRM route used (not fallback)", trip["route_source"] == "osrm", trip["route_source"])
    check("Road route has detailed geometry", len(trip["route_geometry"]["coordinates"]) > 5)

    banner("TEST 4 · Trip fallback (OSRM timeout simulated) + fare rules")
    with mock.patch.object(se._session, "get", side_effect=requests.Timeout("simulated")):
        fb = se.calculate_trip_details(*SITABULDI, *CIVIL_LINES, "Civil Lines")
    _verify_trip(fb, "Fallback")
    hv = se.calculate_distance(*SITABULDI, *CIVIL_LINES)
    check("Fallback = Haversine × 1.3", math.isclose(fb["distance_km"], round(hv * 1.3, 2)),
          f"{hv:.2f} × 1.3")
    check("Minimum fare ₹70 applied for 0.5 km", se.calculate_fare(0.5) == 70.0, f"₹{se.calculate_fare(0.5)}")
    check("10 km fare = ₹190", se.calculate_fare(10) == 190.0, f"₹{se.calculate_fare(10)}")


if __name__ == "__main__":
    for name, fn in [("Haversine", test_haversine), ("Chain", test_chain),
                     ("Landmark", test_landmark), ("Trip", test_trip)]:
        run(name, fn)

    passed = sum(ok for _, ok, _ in results)
    banner(f"SUMMARY · {passed}/{len(results)} checks passed")
    for label, ok, detail in results:
        if not ok:
            print(f"   ❌ {label}  {detail}")
    print("   ✅ READY for Phase 3\n" if passed == len(results) else "   ❌ Fix failures above\n")
    sys.exit(0 if passed == len(results) else 1)
