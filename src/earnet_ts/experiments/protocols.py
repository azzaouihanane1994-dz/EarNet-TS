"""Load preserved protocols and inspect their actual separation."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import tempfile

COLLECTIONS = ("ami", "awe", "earvn10", "uerc2019")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, indent=2, allow_nan=False)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_protocol(root, collection):
    root = Path(root).resolve()
    source = root / "results/reference" / collection / "splits.json"
    saved = json.loads(source.read_text(encoding="utf-8"))
    items = {}
    for split in ("train", "val", "test"):
        rows = []
        for path, label in saved[split]:
            portable = "data/ami/" + Path(path).name if collection == "ami" else path
            absolute = (root / portable).resolve()
            if not absolute.is_relative_to(root / "data") or not absolute.is_file():
                raise ValueError(f"Invalid/missing dataset input: {portable}")
            rows.append((portable, str(label)))
        items[split] = rows
    paths = {s: {row[0] for row in rows} for s, rows in items.items()}
    ids = {s: {row[1] for row in rows} for s, rows in items.items()}
    image_overlap = {a + "/" + b: len(paths[a] & paths[b])
                     for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    id_overlap = {a + "/" + b: len(ids[a] & ids[b])
                  for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    return {
        "collection": collection,
        "dataset_label": "AWE" if collection == "awe" else collection,
        "protocol": "exact_saved_image_partitions",
        "source": source.relative_to(root).as_posix(), "source_sha256": digest(source),
        "items": items, "images": {s: len(v) for s, v in items.items()},
        "identities": {s: len(v) for s, v in ids.items()},
        "image_overlap": image_overlap, "identity_overlap": id_overlap,
        "subject_disjoint": not any(id_overlap.values()),
        "independent_test": not any(image_overlap.values()),
        "preprocessing": "detector_crop_cache" if collection != "ami" else "new_center_square_raw_crop",
        "detector_identity_provenance": "unresolved" if collection != "ami" else "not_used",
    }


def gallery_indices(labels, per_identity=1):
    gallery, probes, seen = [], [], Counter()
    for index, label in enumerate(labels):
        if seen[label] < per_identity:
            gallery.append(index)
        else:
            probes.append(index)
        seen[label] += 1
    if not probes or any(seen[label] <= per_identity for label in seen):
        raise ValueError("Every evaluated identity needs enrollment and at least one probe.")
    return gallery, probes


def audit_protocols(root, output):
    reports = []
    for collection in COLLECTIONS:
        protocol = load_protocol(root, collection)
        report = {k: v for k, v in protocol.items() if k != "items"}
        reports.append(report)
        write_json(Path(output) / "protocols" / f"{collection}.json", protocol)
    report = {
        "protocols": reports,
        "decision": "Preserve original splits; generate separate subject-disjoint protocols.",
        "E01": "Protocols/training are separate; historical AMI overlap is not repaired retrospectively.",
        "detector_identity_provenance": "Detector training identity membership is undocumented.",
    }
    write_json(Path(output) / "protocol_audit.json", report)
    return report


def _pixel_hash(root_and_row):
    from PIL import Image
    root, row = root_and_row
    with Image.open(Path(root) / row[0]) as im:
        rgb = im.convert("RGB")
        content = hashlib.sha256(str(rgb.size).encode("ascii") + rgb.tobytes()).hexdigest()
    return row[0], row[1], content


def prepare_disjoint_protocols(root, output, seeds=(42, 43, 44, 45, 46)):
    import numpy as np
    root, output = Path(root), Path(output)
    reports = {}
    for collection in COLLECTIONS:
        historical = load_protocol(root, collection)
        unique = sorted(set(row for rows in historical["items"].values() for row in rows))
        cache = output / "protocols" / f"{collection}_content_inventory.json"
        inventory_key = hashlib.sha256(json.dumps(unique).encode()).hexdigest()
        if cache.is_file():
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved["membership_sha256"] != inventory_key:
                raise ValueError("Dataset membership changed; use a new experiment output directory.")
            inventory = saved["images"]
        else:
            with ThreadPoolExecutor(max_workers=6) as pool:
                inventory = list(pool.map(_pixel_hash, [(root, row) for row in unique]))
            write_json(cache, {"membership_sha256": inventory_key, "images": inventory})
        by_hash = {}
        for row in inventory:
            by_hash.setdefault(row[2], []).append(row)
        selected, removed, conflicts = [], [], []
        for rows in by_hash.values():
            if len({row[1] for row in rows}) > 1:
                conflicts.extend(rows)
            else:
                selected.append(rows[0]); removed.extend(rows[1:])
        by_id = {}
        for path, label, content in selected:
            by_id.setdefault(label, []).append([path, label, content])
        too_small = {label: rows for label, rows in by_id.items() if len(rows) < 2}
        by_id = {label: rows for label, rows in by_id.items() if len(rows) >= 2}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            identities = np.array(sorted(by_id), dtype=str)
            rng.shuffle(identities)
            ntrain, nval = int(.7 * len(identities)), max(2, int(.1 * len(identities)))
            id_splits = {"train": identities[:ntrain], "val": identities[ntrain:ntrain + nval],
                         "test": identities[ntrain + nval:]}
            items, hashes = {}, {}
            for split, labels in id_splits.items():
                items[split], hashes[split] = [], set()
                for label in labels:
                    rows = list(by_id[label]); rng.shuffle(rows)
                    items[split].extend([row[:2] for row in rows])
                    hashes[split].update(row[2] for row in rows)
            pairs = [("train", "val"), ("train", "test"), ("val", "test")]
            assert not any(set(id_splits[a]) & set(id_splits[b]) or hashes[a] & hashes[b] for a, b in pairs)
            protocol = {
                "collection": collection, "seed": int(seed), "protocol": "custom_subject_disjoint_70_10_20",
                "items": items, "images": {s: len(v) for s, v in items.items()},
                "identities": {s: len(v) for s, v in id_splits.items()},
                "subject_disjoint": True, "independent_test": True,
                "content_overlap": {a + "/" + b: 0 for a, b in pairs},
                "original_split_sha256": historical["source_sha256"],
                "original_split_source": historical["source"],
                "preprocessing": historical["preprocessing"],
                "detector_identity_provenance": historical["detector_identity_provenance"],
                "dataset_label": historical["dataset_label"],
                "official_challenge_protocol": False,
                "gallery_rule": "first manifest image per validation/test identity; remaining images are probes",
                "initialization": "ImageNet ResNet-50 only; no historical ear-transfer checkpoint",
                "duplicate_policy": "drop cross-identity exact RGB conflicts; keep one copy within identity",
            }
            gallery_indices([row[1] for row in items["val"]])
            gallery_indices([row[1] for row in items["test"]])
            write_json(output / "protocols" / f"{collection}_seed{seed}.json", protocol)
        reports[collection] = {"original_unique_images": len(unique), "retained_images": sum(map(len, by_id.values())),
                               "retained_identities": len(by_id), "duplicates_removed": removed,
                               "cross_identity_conflicts_removed": conflicts,
                               "too_small_identities": too_small, "seeds": list(seeds)}
        print(f"Prepared {collection}: {len(by_id)} identities, {sum(map(len, by_id.values()))} distinct images", flush=True)
    write_json(output / "protocol_summary.json", reports)
    return reports
