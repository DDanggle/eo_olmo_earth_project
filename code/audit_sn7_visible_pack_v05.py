#!/usr/bin/env python3
"""Read-only integrity audit of a generated review pack, not human annotation."""
import argparse
import json
import struct
from pathlib import Path

from sn7_visible_pack_v05 import digest, load_pack


def audit(root):
    root = Path(root)
    public = root / "annotator_pack"
    pack = load_pack(public / "pack.json")
    provenance = json.loads((root / "build_provenance.private.json").read_text())
    code_hashes = {name: digest(Path(__file__).with_name(name))
                   for name in provenance["source_code_sha256"]}
    if code_hashes != provenance["source_code_sha256"]:
        raise ValueError("Current source differs from recorded build source")
    dimensions = set()
    for relative in pack["image_sha256"]:
        with (public / relative).open("rb") as handle:
            header = handle.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            raise ValueError(f"Invalid PNG header: {relative}")
        dimensions.add(struct.unpack(">II", header[16:24]))
    episodes = pack["episodes"]
    counts = [len(ep["frames"]) for ep in episodes]
    if len(episodes) != provenance["episode_count"] or sum(counts) != provenance["frame_count"]:
        raise ValueError("Provenance count mismatch")
    return {"pack_id": pack["pack_id"], "split": pack["split"],
            "episodes": len(episodes), "aois": len({ep["aoi"] for ep in episodes}),
            "images": sum(counts), "frames_per_episode": counts,
            "png_dimensions": sorted(dimensions), "image_hashes_verified": True,
            "build_source_hashes_match": True, "human_annotations_produced_by_audit": 0,
            "note": "Integrity only. PNG headers do not establish perceptual readability or valid gold."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack_root")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit(args.pack_root)
    serialized = json.dumps(result, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            handle.write(serialized + "\n")
    print(serialized)
