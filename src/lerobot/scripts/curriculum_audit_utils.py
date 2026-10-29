"""Read-only compatibility checks for curriculum audit recovery.

These helpers do not alter manifests, sampling plans or training configuration.
"""

import ast
import hashlib
import re
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SELF = "src/lerobot/scripts/curriculum_audit_utils.py"
# Only post-training validation bodies may differ. Plan generation, launch
# arguments, model/trainer code and every other source file remain protected.
AUDIT_FUNCTIONS = {
    "src/lerobot/scripts/pi05_ratio_curriculum.py": {"audit"},
    "src/lerobot/scripts/pi05_curriculum_ablation.py": {"check_current_runtime", "audit_suite"},
}


def normalize_path_aliases(reference, candidate):
    """Normalize only differing, existing local directory aliases of the same inode."""
    aliases = []
    for keys in (
        ("dataset_root",),
        ("recipe", "dataset", "root"),
        ("recipe", "policy", "tokenizer_name"),
    ):
        a, b = reference, candidate
        for key in keys[:-1]:
            a, b = a.get(key, {}), b.get(key, {})
        key = keys[-1]
        left, right = a.get(key), b.get(key)
        if left == right or not isinstance(left, str) or not isinstance(right, str):
            continue
        first, second = Path(left), Path(right)
        try:
            same = (
                first.is_absolute()
                and second.is_absolute()
                and first.is_dir()
                and second.is_dir()
                and first.samefile(second)
            )
        except OSError:
            same = False
        if same:
            canonical = str(first.resolve(strict=True))
            a[key] = b[key] = canonical
            aliases.append(
                {"field": ".".join(keys), "reference": left, "candidate": right, "resolved": canonical}
            )
    return aliases


def differences(reference, candidate, prefix=""):
    """Return leaf-level differences instead of an uninformative 'recipe' error."""
    if isinstance(reference, dict) and isinstance(candidate, dict):
        rows = []
        for key in sorted(reference.keys() | candidate.keys()):
            rows.extend(
                differences(
                    reference.get(key, "<missing>"),
                    candidate.get(key, "<missing>"),
                    f"{prefix}.{key}" if prefix else key,
                )
            )
        return rows
    return (
        [] if reference == candidate else [{"field": prefix, "reference": reference, "candidate": candidate}]
    )


def _git(root, *args, input_data=None):
    return subprocess.run(
        ["git", *args], cwd=root, input=input_data, capture_output=True, check=True, timeout=30
    ).stdout


@lru_cache(maxsize=16)
def committed_sources(root, commit):
    """Rebuild the exact path/content digest used by PDETStreams.runtime_info()."""
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("Missing/invalid recorded commit; cannot verify audit-only compatibility")
    listing = _git(root, "ls-tree", "-r", "-z", commit, "--", "src/lerobot", "examples/training")
    entries = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        meta, name = entry.split(b"\t", 1)
        mode, kind, oid = meta.decode().split()
        name = name.decode()
        if (name.startswith("src/lerobot/") and name.endswith(".py")) or (
            name.startswith("examples/training/") and name.endswith(".sh")
        ):
            if kind != "blob" or mode not in ("100644", "100755"):
                raise ValueError(f"Unsupported source entry: {name}")
            entries.append((name, oid))
    if not entries:
        raise ValueError("Recorded commit has no training sources")
    payload = _git(root, "cat-file", "--batch", input_data="".join(f"{oid}\n" for _, oid in entries).encode())
    sources, offset = {}, 0
    for name, oid in entries:
        end = payload.index(b"\n", offset)
        actual_oid, kind, size = payload[offset:end].decode().split()
        if actual_oid != oid or kind != "blob":
            raise ValueError(f"Cannot read recorded source: {name}")
        offset = end + 1
        sources[name] = payload[offset : offset + int(size)]
        offset += int(size) + 1
    ordered = sorted(p for p in sources if p.startswith("src/lerobot/")) + sorted(
        p for p in sources if p.startswith("examples/training/")
    )
    digest = hashlib.sha256()
    for path in ordered:
        digest.update(path.encode() + b"\0")
        digest.update(hashlib.sha256(sources[path]).digest())
    return sources, digest.hexdigest()


def _training_sources(sources):
    result = {}
    for path, content in sources.items():
        if path == SELF:
            continue  # Imported only from the audit bodies enumerated above.
        if path in AUDIT_FUNCTIONS:
            tree = ast.parse(content)
            tree.body = [
                node
                for node in tree.body
                if not (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name in AUDIT_FUNCTIONS[path]
                )
            ]
            content = ast.dump(tree, include_attributes=False).encode()
        result[path] = content
    return result


def compare_runtime(reference, candidate, *, repo_root=ROOT):
    """Allow only verified audit edits; reject environment drift and dirty source hashes."""
    for key in ("versions", "cuda"):
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"Different code/environment: {key}")
    code_keys = ("git_commit", "source_tree_sha256")
    records = [{key: record.get(key) for key in code_keys} for record in (reference, candidate)]
    if records[0] == records[1]:
        return {"status": "same_recorded_code", "records": records}
    versions = []
    try:
        for record in records:
            sources, digest = committed_sources(str(Path(repo_root).resolve()), record["git_commit"])
            if digest != record["source_tree_sha256"]:
                raise ValueError(
                    "Recorded on-disk source hash differs from its Git tree; historical or current "
                    "uncommitted changes prevent audit-only compatibility verification"
                )
            versions.append(sources)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Cannot inspect recorded Git history locally; fetch the recorded commits") from error
    protected = differences(_training_sources(versions[0]), _training_sources(versions[1]))
    if protected:
        raise ValueError(f"Training-relevant source changes: {[row['field'] for row in protected]}")
    return {
        "status": "audit_only_changes_verified",
        "records": records,
        "changed_source_paths": [row["field"] for row in differences(*versions)],
        "scope": "Both recorded source hashes match their Git trees; training/plan/launcher code is "
        "unchanged outside named audit bodies and their helper. Not numerical determinism.",
    }
