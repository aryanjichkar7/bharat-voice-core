#!/usr/bin/env python3
"""
Bharat Online Cabs — Idempotent OpenStreetMap Extract Importer CLI.

Imports a regional OpenStreetMap extract (.osm.pbf or .osm XML) into the local SQLite
database with FTS5 and RTree spatial indexes.

Usage:
    python import_osm_places.py --input data/nagpur.osm.pbf
    python import_osm_places.py --input data/nagpur.osm
    python import_osm_places.py --info
"""
import argparse
import sys
from pathlib import Path

from osm_indexer import (
    DEFAULT_DB_PATH,
    DEFAULT_SERVICE_BBOX,
    OSMPlaceImporter,
    get_index_stats,
    OSM_ATTRIBUTION,
    OSM_LICENSE_URL,
)


def print_stats(db_path: str):
    stats = get_index_stats(db_path)
    print("\n" + "=" * 60)
    print("Bharat Online Cabs — Local OSM Place Index Status")
    print("=" * 60)
    print(f"Database Path : {stats['database_path']}")
    print(f"Exists        : {stats['database_exists']}")
    if stats["database_exists"]:
        print(f"File Size     : {stats['size_mb']} MB ({stats['size_bytes']:,} bytes)")
        print(f"Total Places  : {stats['total_rows']:,}")
        print("\nRow Counts by Feature Category:")
        for cat, cnt in sorted(stats["categories"].items(), key=lambda x: -x[1]):
            print(f"  * {cat:15}: {cnt:,}")
        if stats.get("metadata"):
            m = stats["metadata"]
            print("\nProvenance & Extract Metadata:")
            print(f"  * Source File : {m.get('source_file')}")
            print(f"  * Import Time : {m.get('import_timestamp')}")
            print(f"  * Duration    : {m.get('import_duration_seconds')}s")
            print(f"  * Bounding Box: {m.get('bbox')}")
        print(f"\nAttribution   : {OSM_ATTRIBUTION}")
        print(f"License       : {OSM_LICENSE_URL}")
    else:
        print("\n[!] Local place index has not been built yet.")
        print("To generate the index:")
        print("1. Obtain a Nagpur-area OpenStreetMap extract (.osm.pbf or .osm XML).")
        print("   - From BBBike: https://extract.bbbike.org/ (BBox: 78.85, 20.92, 79.33, 21.37)")
        print("   - Or place custom extract at: data/nagpur.osm.pbf")
        print("2. Run: python import_osm_places.py --input data/nagpur.osm.pbf")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Idempotent OSM Extract Importer for Bharat Online Cabs")
    parser.add_argument("--input", "-i", type=str, help="Path to .osm.pbf or .osm XML extract file")
    parser.add_argument("--db", "-d", type=str, default=DEFAULT_DB_PATH, help=f"Target SQLite database (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--info", action="store_true", help="Print current database status and row counts")
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_SERVICE_BBOX,
                        help="Bounding box (min_lat, min_lon, max_lat, max_lon)")

    args = parser.parse_args()

    if args.info or not args.input:
        print_stats(args.db)
        if not args.input:
            return 0

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    importer = OSMPlaceImporter(db_path=args.db, bbox=tuple(args.bbox))
    suffix = input_path.suffix.lower()

    print(f"[*] Processing extract: {input_path}")
    print(f"[*] Target database   : {args.db}")
    print(f"[*] Service BoundingBox: {args.bbox}")

    if suffix == ".pbf":
        meta = importer.import_pbf(input_path)
    elif suffix in (".osm", ".xml"):
        meta = importer.import_xml(input_path)
    else:
        print(f"Error: Unsupported file format {suffix}. Expected .pbf or .osm/.xml", file=sys.stderr)
        return 1

    print("\n[+] Import successfully completed!")
    print_stats(args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
