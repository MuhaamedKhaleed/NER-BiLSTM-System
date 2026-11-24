"""Explicit source download. No import-time network or data redistribution."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="New source directory"
    )
    parser.add_argument(
        "--cached", type=Path, help="Use already downloaded files, without network"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for filename, url, sha in SOURCES:
        target = args.output / filename
        if args.cached:
            shutil.copyfile(args.cached / filename, target)
        else:
            with (
                urllib.request.urlopen(url, timeout=120) as source,
                target.open("wb") as out,
            ):
                shutil.copyfileobj(source, out)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != sha:
            target.unlink()
            raise ValueError("Source checksum mismatch: " + filename)
        manifest.append({"file": filename, "url": url, "sha256": actual})
    (args.output / "source-manifest.json").write_text(json.dumps(manifest, indent=2))


SOURCES = [
    (
        "wnut17train.conll",
        "https://raw.githubusercontent.com/leondz/emerging_entities_17/e52a2d2a71ac1a051ca2d532eef28653c5c02603/wnut17train.conll",
        "731820e13f71af324c6b55a1575ec2ce59fbaa2a0806f8f0400b98d56cd6a7a5",
    ),
    (
        "emerging.dev.conll",
        "https://raw.githubusercontent.com/leondz/emerging_entities_17/e52a2d2a71ac1a051ca2d532eef28653c5c02603/emerging.dev.conll",
        "e053b752b8155113ca66e7db08af5f2f5f43ab993694e4774f1ec5bad42312d8",
    ),
    (
        "README.md",
        "https://raw.githubusercontent.com/leondz/emerging_entities_17/e52a2d2a71ac1a051ca2d532eef28653c5c02603/README.md",
        "7dba67f5d6f7789af3b36af9cbea54b73705f86df100002ae202a9c2f5750157",
    ),
]

if __name__ == "__main__":
    main()
