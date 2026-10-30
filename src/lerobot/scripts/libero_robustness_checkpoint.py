"""Scope-aware checkpoint validation without changing the benchmark protocol hash."""

import argparse

from lerobot.scripts.libero_vlm_only import validate_checkpoint

SCOPES = ("joint", "expert_only", "prompt_only")


def check(directory, scope="joint"):
    if scope not in SCOPES:
        raise ValueError(f"Invalid training scope: {scope}")
    config = validate_checkpoint(directory, 2, train_action_expert=scope != "prompt_only")
    frozen = config.get("freeze_vlm_prompt", False)
    if type(frozen) is not bool or frozen != (scope == "expert_only"):
        raise ValueError(f"freeze_vlm_prompt disagrees with TRAIN_SCOPE={scope}")
    return {"checkpoint": "passed", "scope": scope, "vlm_tokens": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--scope", choices=SCOPES, default="joint")
    args = parser.parse_args()
    check(args.directory, args.scope)
    print(f"Checkpoint: VLM2 / {args.scope}; saved weights and processors verified")


if __name__ == "__main__":
    main()
