from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


CLASSES = {"bird_drop", "clean", "crack_or_damage", "dust", "snow"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
REQUIRED_FIELDS = {
    "id",
    "expected_class",
    "source_page",
    "image_url",
    "license",
    "author",
}


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_entries(manifest_path: Path) -> list[dict]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data["images"] if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError("Manifest bir liste veya 'images' listesi iceren nesne olmali.")
    return entries


def known_dataset_hashes(project_root: Path) -> set[str]:
    roots = [
        project_root / "datasets/processed/classification_5class",
        project_root / "datasets/processed/detection_solar_panel_yolo",
    ]
    hashes: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                hashes.add(sha1_file(path))
    return hashes


def download(entry: dict, output_root: Path) -> Path:
    missing = REQUIRED_FIELDS - entry.keys()
    if missing:
        raise ValueError(f"{entry.get('id', '?')}: eksik alanlar: {sorted(missing)}")
    if entry["expected_class"] not in CLASSES:
        raise ValueError(f"{entry['id']}: gecersiz sinif.")
    if not str(entry["license"]).strip():
        raise ValueError(f"{entry['id']}: lisans bilgisi bos olamaz.")

    suffix = Path(urlparse(entry["image_url"]).path).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".jpg"
    destination = output_root / entry["expected_class"] / f"{entry['id']}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    request = urllib.request.Request(
        entry["image_url"],
        headers={"User-Agent": "SolarPanelThesisDemo/1.0"},
    )
    for attempt, delay in enumerate((0, 3, 8, 15), start=1):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    raise ValueError(f"{entry['id']}: URL bir goruntu dondurmedi.")
                destination.write_bytes(response.read())
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
    time.sleep(1)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manifestteki lisansli harici test fotograflarini indirir."
    )
    parser.add_argument(
        "--manifest",
        default="demo_samples/external/manifest.json",
    )
    parser.add_argument(
        "--output",
        default="demo_samples/external",
    )
    parser.add_argument(
        "--skip-dataset-hash-check",
        action="store_true",
        help="Buyuk veri klasorlerinde tam SHA-1 taramasini atlar.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    manifest_path = (project_root / args.manifest).resolve()
    output_root = (project_root / args.output).resolve()
    entries = load_entries(manifest_path)
    dataset_hashes = (
        set() if args.skip_dataset_hash_check else known_dataset_hashes(project_root)
    )

    kept_entries = []
    for entry in entries:
        existing_path = (
            project_root / entry["local_path"] if entry.get("local_path") else None
        )
        path = (
            existing_path
            if existing_path is not None and existing_path.exists()
            else download(entry, output_root)
        )
        digest = sha1_file(path)
        if digest in dataset_hashes:
            path.unlink(missing_ok=True)
            print(f"CIKARILDI veri sizintisi: {entry['id']}")
            continue
        entry["sha1"] = digest
        entry["local_path"] = path.relative_to(project_root).as_posix()
        kept_entries.append(entry)
        print(f"INDIRILDI {entry['id']} -> {entry['local_path']}")

    manifest_path.write_text(
        json.dumps({"images": kept_entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Tamamlandi: {len(kept_entries)} fotograf")


if __name__ == "__main__":
    main()
