"""Real symlink and Git-tree regressions for post-training audit recovery."""

import copy
import subprocess

import pytest

from lerobot.scripts import curriculum_audit_utils as audit


def test_dataset_and_tokenizer_aliases_are_normalized_without_hiding_other_changes(tmp_path):
    real, alias = tmp_path / "real", tmp_path / "alias"
    real.mkdir()
    alias.symlink_to(real, target_is_directory=True)
    old = {
        "dataset_root": str(alias),
        "recipe": {
            "dataset": {"root": str(alias)},
            "policy": {"tokenizer_name": str(alias)},
            "batch_size": 32,
        },
    }
    new = {
        "dataset_root": str(real),
        "recipe": {"dataset": {"root": str(real)}, "policy": {"tokenizer_name": str(real)}, "batch_size": 64},
    }
    report = audit.normalize_path_aliases(old, new)
    assert len(report) == 3
    assert audit.differences(old, new) == [{"field": "recipe.batch_size", "reference": 32, "candidate": 64}]


@pytest.mark.parametrize("kind", ["different", "missing", "hub"])
def test_different_or_missing_directories_and_hub_names_are_not_equated(tmp_path, kind):
    one, two = tmp_path / "one", tmp_path / "two"
    if kind == "different":
        one.mkdir()
        two.mkdir()
    if kind == "hub":
        one, two = "google/tokenizer", str(two)
    old, new = (
        {"recipe": {"policy": {"tokenizer_name": str(one)}}},
        {"recipe": {"policy": {"tokenizer_name": str(two)}}},
    )
    assert audit.normalize_path_aliases(old, new) == []
    assert audit.differences(old, new)[0]["field"] == "recipe.policy.tokenizer_name"


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root).decode().strip()


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "Test")
    paths = {
        "src/lerobot/scripts/pi05_ratio_curriculum.py": "def make_plan():\n    return 42\n\ndef audit():\n    return 'old'\n",
        "src/lerobot/scripts/pi05_curriculum_ablation.py": "def build_jobs():\n    return 42\n\ndef check_current_runtime():\n    return 'old'\n"
        "\ndef audit_suite():\n    return 'old'\n",
        "src/lerobot/policies/model.py": "WEIGHT = 1\n",
        "examples/training/train.sh": "echo train\n",
    }
    for name, text in paths.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "initial")
    return tmp_path


def record(root):
    sha = git(root, "rev-parse", "HEAD")
    return {
        "git_commit": sha,
        "source_tree_sha256": audit.committed_sources(str(root), sha)[1],
        "versions": {"torch": "test"},
        "cuda": "test",
    }


def test_committed_digest_matches_independent_on_disk_algorithm(repository):
    import hashlib

    digest = hashlib.sha256()
    paths = sorted((repository / "src/lerobot").rglob("*.py")) + sorted(
        (repository / "examples/training").rglob("*.sh")
    )
    for path in paths:
        digest.update(str(path.relative_to(repository)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    assert record(repository)["source_tree_sha256"] == digest.hexdigest()


def test_only_audit_body_changes_are_compatible(repository):
    old = record(repository)
    for name in audit.AUDIT_FUNCTIONS:
        path = repository / name
        path.write_text(path.read_text().replace("return 'old'", "return 'fixed'"))
    (repository / audit.SELF).write_text("# audit helpers only\n")
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "audit repair")
    new = record(repository)
    assert audit.compare_runtime(old, new, repo_root=repository)["status"] == "audit_only_changes_verified"
    assert audit.compare_runtime(new, old, repo_root=repository)["status"] == "audit_only_changes_verified"
    dirty = copy.deepcopy(new)
    dirty["source_tree_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="uncommitted"):
        audit.compare_runtime(old, dirty, repo_root=repository)
    with pytest.raises(ValueError, match="environment"):
        audit.compare_runtime(old, {**new, "cuda": "other"}, repo_root=repository)


@pytest.mark.parametrize(
    "path",
    [
        "src/lerobot/scripts/pi05_ratio_curriculum.py",
        "src/lerobot/scripts/pi05_curriculum_ablation.py",
        "src/lerobot/policies/model.py",
        "examples/training/train.sh",
    ],
)
def test_training_plan_launcher_and_model_changes_are_rejected(repository, path):
    old = record(repository)
    target = repository / path
    target.write_text(target.read_text().replace("42", "43").replace("1", "2").replace("echo", "printf"))
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "training changed")
    with pytest.raises(ValueError, match="Training-relevant"):
        audit.compare_runtime(old, record(repository), repo_root=repository)
