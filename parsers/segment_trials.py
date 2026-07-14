"""parsers.segment_trials — data/raw -> data/processed atomic samples + index.

THE single entrypoint that realises the BFId lab ``data/raw -> data/processed``
transformation (Todt/Morsbach/Strufe, CCS '25). It walks the hive-partitioned
raw tree::

    data/raw/participant=P001/style=normal/trial=001/
        metadata.json  csi_p1.raw ... csi_p4.raw  bfi_recorder.pcapng  logs/

reads each trial's ``metadata.json`` (canonical ``contract.TrialMetadata``
schema) and, for every active perspective and modality, produces the atomic
sample(s):

  * CSI: each ``csi_p<n>.raw`` -> one ``..._csi_p<n>.npz`` via parse_csi.
  * BFI: the central ``bfi_recorder.pcapng`` -> one ``..._bfi_p<n>.npz`` PER
    active perspective via parse_bfi (the recorder captures all perspectives in
    one pcapng; per-perspective separation is the parser's job).

Each sample is written under ``data/processed/samples/`` with the canonical npz
schema, and one row per sample is APPENDED to ``data/processed/samples.parquet``
(the index models/dataset.py reads). Re-running updates rows for regenerated
samples rather than clobbering the whole index.

Index columns (one row per sample)::

    sample, path, label, style, perspective, modality,
    n_timesteps, n_features, source_file, trial, ts_utc

Parquet is written via pandas/pyarrow when available (``[ml]`` extra). If they
are absent the index is written as a CSV sidecar (``samples.parquet.csv``) and a
JSON-lines sidecar so the pipeline still runs on a partially-provisioned box; a
note is logged. numpy is always required.

CLI::

    python3 -m parsers.segment_trials --data-root data --participant P001 \
        [--style normal] [--trial 001]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wallflower import contract

from . import parse_bfi, parse_csi

AGENT = "segment_trials"
log = logging.getLogger(AGENT)

# Optional dataframe stack (only needed to write real parquet) ----------------
try:  # pragma: no cover - depends on [ml] extra
    import pandas as pd  # type: ignore

    _HAVE_PANDAS = True
except Exception:  # noqa: BLE001
    pd = None  # type: ignore
    _HAVE_PANDAS = False

INDEX_COLUMNS = (
    "sample", "path", "label", "style", "perspective", "modality",
    "n_timesteps", "n_features", "source_file", "trial", "ts_utc",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Raw tree discovery
# --------------------------------------------------------------------------- #
def _partition_value(part: str, key: str) -> str | None:
    prefix = f"{key}="
    return part[len(prefix):] if part.startswith(prefix) else None


def discover_trials(data_root: Path | str, *, participant: str | None = None,
                    style: str | None = None, trial: str | None = None
                    ) -> list[tuple[str, str, str, Path]]:
    """Find (participant, style, trial, trial_dir) tuples under data/raw.

    Optional filters narrow the walk. Only directories that actually contain a
    ``metadata.json`` are returned.
    """
    raw_root = Path(data_root) / "raw"
    out: list[tuple[str, str, str, Path]] = []
    if not raw_root.is_dir():
        return out
    for p_dir in sorted(raw_root.glob("participant=*")):
        pid = _partition_value(p_dir.name, "participant")
        if pid is None or (participant and pid != participant):
            continue
        for s_dir in sorted(p_dir.glob("style=*")):
            sty = _partition_value(s_dir.name, "style")
            if sty is None or (style and sty != style):
                continue
            for t_dir in sorted(s_dir.glob("trial=*")):
                tri = _partition_value(t_dir.name, "trial")
                if tri is None or (trial and tri != trial):
                    continue
                if (t_dir / contract.metadata_name()).is_file():
                    out.append((pid, sty, tri, t_dir))
    return out


def _read_metadata(trial_dir: Path) -> dict[str, Any]:
    path = trial_dir / contract.metadata_name()
    return json.loads(path.read_text(encoding="utf-8"))


def _active_perspectives(meta: dict[str, Any]) -> list[int]:
    persp = meta.get("perspectives") or []
    out = [int(p) for p in persp if int(p) in contract.PERSPECTIVES]
    return out or list(contract.PERSPECTIVES)


# --------------------------------------------------------------------------- #
# Per-trial segmentation
# --------------------------------------------------------------------------- #
def segment_trial(data_root: Path | str, participant: str, style: str, trial: str,
                  trial_dir: Path) -> list[dict[str, Any]]:
    """Produce all atomic samples for one trial. Returns the sample-info rows.

    A capture that cannot be decoded is logged and skipped (no placeholder is
    written), so one undecodable perspective doesn't abort the whole session.
    """
    meta = _read_metadata(trial_dir)
    perspectives = _active_perspectives(meta)
    rows: list[dict[str, Any]] = []

    # CSI: one sample per per-perspective .raw that exists.
    for p in perspectives:
        raw = trial_dir / contract.csi_raw_name(p)
        if not raw.exists():
            continue
        try:
            info = parse_csi.parse_csi(raw, p, participant, style, trial,
                                       data_root=data_root)
        except RuntimeError as exc:
            log.warning("skipping CSI perspective %d in %s: %s", p, trial_dir, exc)
            continue
        info["trial"] = trial
        info["ts_utc"] = _now_iso()
        rows.append(info)

    # BFI: one sample per perspective from the single central recorder pcapng.
    pcap = trial_dir / contract.bfi_recorder_name()
    if pcap.exists():
        for p in perspectives:
            try:
                info = parse_bfi.parse_bfi(pcap, p, participant, style, trial,
                                           data_root=data_root)
            except RuntimeError as exc:
                log.warning("skipping BFI perspective %d in %s: %s", p, trial_dir, exc)
                continue
            info["trial"] = trial
            info["ts_utc"] = _now_iso()
            rows.append(info)

    return rows


# --------------------------------------------------------------------------- #
# Index (parquet) — append, don't clobber; update rows for regenerated samples
# --------------------------------------------------------------------------- #
def _index_path(data_root: Path | str) -> Path:
    return Path(data_root) / "processed" / contract.SAMPLES_PARQUET


def _csv_sidecar(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(parquet_path.suffix + ".csv")


def _jsonl_sidecar(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(parquet_path.suffix + ".jsonl")


def _row_for_index(info: dict[str, Any]) -> dict[str, Any]:
    return {col: info.get(col) for col in INDEX_COLUMNS}


def _load_existing_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def update_index(data_root: Path | str, new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge ``new_rows`` into the samples index (keyed by ``sample``).

    Always maintains a JSON-lines sidecar (stdlib, lossless) as the source of
    truth for merging, then materialises ``samples.parquet`` via pandas if
    available else a CSV sidecar. Existing rows for the same ``sample`` are
    replaced (idempotent re-runs), others preserved (append, don't clobber).
    """
    parquet_path = _index_path(data_root)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = _jsonl_sidecar(parquet_path)

    merged: dict[str, dict[str, Any]] = {}
    for r in _load_existing_jsonl(jsonl_path):
        if r.get("sample"):
            merged[r["sample"]] = _row_for_index(r)
    for info in new_rows:
        row = _row_for_index(info)
        merged[row["sample"]] = row

    ordered = [merged[k] for k in sorted(merged)]

    # JSON-lines source of truth (stdlib).
    jsonl_path.write_text(
        "".join(json.dumps(r) + "\n" for r in ordered), encoding="utf-8")

    result: dict[str, Any] = {
        "index_jsonl": str(jsonl_path),
        "total_rows": len(ordered),
        "new_or_updated": len(new_rows),
    }

    if _HAVE_PANDAS:
        df = pd.DataFrame(ordered, columns=list(INDEX_COLUMNS))
        try:
            df.to_parquet(parquet_path, index=False)
            result["index_parquet"] = str(parquet_path)
            result["parquet_backend"] = "pandas"
        except Exception as exc:  # noqa: BLE001 - missing pyarrow/fastparquet
            csv_path = _csv_sidecar(parquet_path)
            df.to_csv(csv_path, index=False)
            result["index_csv"] = str(csv_path)
            result["parquet_backend"] = "csv_fallback"
            result["parquet_error"] = str(exc)
    else:
        # No pandas: write a CSV sidecar via stdlib csv.
        import csv

        csv_path = _csv_sidecar(parquet_path)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(INDEX_COLUMNS))
            w.writeheader()
            for r in ordered:
                w.writerow(r)
        result["index_csv"] = str(csv_path)
        result["parquet_backend"] = "csv_fallback_no_pandas"

    return result


# --------------------------------------------------------------------------- #
# Top-level run
# --------------------------------------------------------------------------- #
def run(data_root: Path | str, *, participant: str | None = None,
        style: str | None = None, trial: str | None = None) -> dict[str, Any]:
    """Segment all matching trials and update the index. Returns a summary dict."""
    trials = discover_trials(data_root, participant=participant, style=style,
                             trial=trial)
    all_rows: list[dict[str, Any]] = []
    per_trial: list[dict[str, Any]] = []
    for pid, sty, tri, tdir in trials:
        rows = segment_trial(data_root, pid, sty, tri, tdir)
        all_rows.extend(rows)
        per_trial.append({
            "participant": pid, "style": sty, "trial": tri,
            "trial_dir": str(tdir), "samples": len(rows),
        })

    index = update_index(data_root, all_rows) if all_rows else {
        "index_jsonl": str(_jsonl_sidecar(_index_path(data_root))),
        "total_rows": 0, "new_or_updated": 0,
    }

    summary = {
        "agent": AGENT,
        "ok": True,
        "ts_utc": _now_iso(),
        "data_root": str(data_root),
        "filters": {"participant": participant, "style": style, "trial": trial},
        "trials_found": len(trials),
        "samples_written": len(all_rows),
        "pandas_available": _HAVE_PANDAS,
        "index": index,
        "per_trial": per_trial,
    }
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"parsers.{AGENT}",
        description="Walk data/raw, parse each active perspective+modality into "
                    "ML-ready npz samples and append the samples index. The "
                    "single data/raw -> data/processed entrypoint.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--participant", help="filter to one participant (e.g. P001)")
    p.add_argument("--style", help="filter to one walking style")
    p.add_argument("--trial", help="filter to one trial id (e.g. 001)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = run(args.data_root, participant=args.participant, style=args.style,
                  trial=args.trial)
    print(json.dumps(summary))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
