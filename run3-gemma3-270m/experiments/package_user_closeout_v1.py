"""Package Dorian's explicit early closeout; no models, experiments, or scores.

Unlike the earlier unused packaging helpers, this honors the final instruction
to skip verify.py and GSM8K, include the nonpassing best attempt and downloads,
and supply the original model separately as split files.
"""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import os
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "export"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def folder_bytes():
    total = 0
    for directory, subdirs, files in os.walk(ROOT, followlinks=False):
        subdirs[:] = [d for d in subdirs if not (Path(directory) / d).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                total += path.stat().st_size
    return total


assert (ROOT / "REPORT.md").read_text().splitlines()[0].endswith("— final report")
closeout = json.loads((ROOT / "artifacts/final_user_closeout.json").read_text())
assert closeout["verify_py"] == "skipped at explicit user instruction"
assert closeout["gsm8k"] == "skipped at explicit user instruction"
assert not list((ROOT / "best").glob("H[1-5]/delta.pt"))
assert not (ROOT / "verification/results.json").exists()
assert not (EXPORT / "run3.zip").exists(), "Never overwrite the final archive"
assert not (EXPORT / "run3.zip.partial").exists()

head = "0" * 64
ledger = []
with (ROOT / "ledger.jsonl").open("rb") as stream:
    for raw in stream:
        row = json.loads(raw)
        assert row["previous_sha256"] == head
        head = hashlib.sha256(raw).hexdigest()
        ledger.append(row)
assert head == closeout["ledger_head_sha256"]
assert sha(ROOT / "ledger.jsonl") == closeout["ledger_file_sha256"]
assert {r["run_id"] for r in ledger if r["event"] == "start"} == {
    r["run_id"] for r in ledger if r["event"] == "finish"
}

score = json.loads((ROOT / "best/H1-best-attempt/score.json").read_text())
assert not score["passed"]["H1"]
assert sha(ROOT / "best/H1-best-attempt/delta.pt") == score["checkpoint_sha256"]
for target, original in (
    ("delta.pt", "experiments/a313v1_duplicate_reference_compression/delta.pt"),
    ("delta.manifest.json", "experiments/a313v1_duplicate_reference_compression/delta.manifest.json"),
    ("score.json", "scores/a313v1_full_H1.json"),
    ("edit_categories.json", "artifacts/a313v1_duplicate_reference_compression_edit_category_audit.json"),
):
    assert sha(ROOT / "best/H1-best-attempt" / target) == sha(ROOT / original)

members = {}


def add(path):
    assert path.is_file() and not path.is_symlink(), path
    name = path.relative_to(ROOT).as_posix()
    assert not name.startswith("/") and ".." not in Path(name).parts
    assert name != "downloads/model.safetensors"
    if path.suffix in {".pt", ".safetensors", ".npy", ".npz", ".pkl", ".pickle"}:
        assert name == "best/H1-best-attempt/delta.pt", name
    members[name] = path


for name in ("GOAL.md", "REPORT.md", "NOTES.md", "ledger.jsonl"):
    add(ROOT / name)

script_suffixes = {".py", ".sh", ".bash", ".js", ".mjs", ".cjs"}
for directory in ("tools", "experiments", "prior", "artifacts", "scores", "host"):
    for path in sorted((ROOT / directory).rglob("*")):
        if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
            continue
        if (
            directory == "tools"
            or path.suffix in script_suffixes
            or "manifest" in path.name.lower()
            or path.name == "frozen.json"
            or (path.suffix == ".json" and directory in {"experiments", "artifacts", "scores"})
            or (directory == "prior" and path.suffix == ".md")
        ):
            add(path)

# The interrupted analysis has no completed JSON; preserve its partial log.
add(ROOT / "artifacts/a362v1_coupled_native_feature_output.log")
for path in sorted((ROOT / "best").rglob("*")):
    if path.is_file() and "__pycache__" not in path.parts:
        add(path)
for path in sorted((ROOT / "downloads").rglob("*")):
    if path.is_file() and path != ROOT / "downloads/model.safetensors":
        add(path)
for path in ROOT.iterdir():
    if path.is_file() and (path.suffix in script_suffixes or "manifest" in path.name.lower()):
        add(path)

assert "experiments/package_user_closeout_v1.py" in members
assert "best/H1-best-attempt/delta.pt" in members
assert "downloads/tokenizer.json" in members
assert all(not n.startswith(("heldout/", "calib/", ".venv/")) for n in members)
assert all(not n.startswith("artifacts/") or Path(n).suffix in {".json", ".py", ".sh", ".bash", ".log"} for n in members)

payload = {name: {"bytes": path.stat().st_size, "sha256": sha(path)} for name, path in sorted(members.items())}
metadata = {
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "reason": "Explicit user-directed early closeout; no new experiment, build, or score",
    "leader": "A313", "leader_status": "NOT PASSING H1", "banked_rungs": [],
    "leader_checkpoint_sha256": score["checkpoint_sha256"],
    "ledger_head_sha256": head, "ledger_file_sha256": closeout["ledger_file_sha256"],
    "meter_used": closeout["meter_used"], "meter_remaining": closeout["meter_remaining"],
    "verify_py": "skipped at user request", "gsm8k": "skipped at user request",
    "independent_retest": "pending", "activation_artifacts_included": False,
    "full_source_model_inside_zip": False, "files": payload,
}

# Reserve room for the uncompressed archive upper bound, manifests, and shards.
uncompressed = sum(x["bytes"] for x in payload.values())
projected = folder_bytes() + uncompressed + (ROOT / "downloads/model.safetensors").stat().st_size + 10_000_000
assert projected < 40_000_000_000, projected
print(json.dumps({"selected_files": len(payload), "uncompressed_bytes": uncompressed,
                  "projected_folder_bytes": projected}), flush=True)

temporary = EXPORT / "run3.zip.partial"
with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for name, path in sorted(members.items()):
        archive.write(path, name)
    archive.writestr("PACKAGE_MANIFEST.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n")
assert temporary.stat().st_size < 300_000_000
with zipfile.ZipFile(temporary) as archive:
    assert archive.testzip() is None
    assert set(archive.namelist()) == set(payload) | {"PACKAGE_MANIFEST.json"}
    for name, expected in payload.items():
        digest = hashlib.sha256()
        with archive.open(name) as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == expected["sha256"], name
os.replace(temporary, EXPORT / "run3.zip")
metadata.update(archive="export/run3.zip", archive_bytes=(EXPORT / "run3.zip").stat().st_size,
                archive_sha256=sha(EXPORT / "run3.zip"), archive_crc_and_all_member_hashes="passed")
(EXPORT / "package_manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
assert sha(ROOT / "ledger.jsonl") == closeout["ledger_file_sha256"]
print(json.dumps({k: v for k, v in metadata.items() if k != "files"}), flush=True)
