# Real-data CPU development run

Source: https://github.com/leondz/emerging_entities_17 . Actual recorded result: `real-data-output.json`. This is a small development run, not full training, production readiness or an untouched benchmark. No data/model artifacts are committed. Run from the repository root after installing the locked ML/dev extras. This script starts real CPU training only when explicitly invoked. Outputs use fixed new names under the supplied data folder and refuse rerunning into an existing artifact. Use a new folder for another run.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/real_data.py --data /path/to/prepared-source
```

Clone/download upstream at https://github.com/leondz/emerging_entities_17 with README attribution/CC BY4.0 terms. Put `wnut17train.conll` and `emerging.dev.conll` in the supplied source directory. Script reads complete sentences, first512 official train and128 official dev; final test unused. Token accuracy is not entity-span F1. Social-media text can contain identifiers, abusive language and source privacy restrictions. No text examples or trained artifacts are published. Source commit during run: e52a2d2a71ac1a051ca2d532eef28653c5c02603.

## Actual limits

small2epoch CPU development run; token accuracy, not entity-span F1; social-text provenance/privacy limits; not production NER

## Explicit download and preparation

The optional helper downloads only the documented primary-source files, checks SHA-256 and refuses an existing destination. It never runs during installation, import or tests. Source attribution/license cautions above still apply. No model/data is bundled. For an offline copy, add `--cached /path/to/source-files` with the same filenames; checksums are still required.

```sh
uv run --no-sync python scripts/fetch_real_data.py --output ./real-source
uv run --no-sync python scripts/real_data.py --data ./real-source
```

The hashes pin the exact small development sources used here, not a permission warranty or a quality benchmark. If a source changes, the helper fails rather than silently training on different data.

## Docker without a registry account

This repository supplies a Dockerfile, not a published registry image. Build locally once. The default remains the small invented-input wiring demo. No account, paid service or automatic source download is required by the image.

```sh
docker build -f Dockerfile.ml -t project-ml .
docker run --rm --network none project-ml
```

The image also includes `/app/scripts`. After explicit source preparation on the host, mount a new writable working directory and invoke the real script with `--entrypoint python`. Keep model/source mounts read-only when the script does not write beside them. The image uses uid10001, so grant that uid write access to the output folder, do not run the container as root to bypass this. Data/preprocessing scripts currently place artifacts beside source files; mount the working copy read-write and keep your original untouched. The environment disables Hugging Face downloads. Network-none is suitable only after sources/models are local; an explicit fetch helper needs network by design. CPU/dependency builds were checked on GitHub's Linux runner, not every laptop OS.
