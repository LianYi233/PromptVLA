"""Export/check a tiny step-zero prompt for an initialization-swap experiment."""

import argparse
import json

from lerobot.utils.prompt_init_names import rename_libero_prompts, resolve_libero_prompt
from lerobot.utils.prompt_initialization import (
    export_initial_prompt,
    read_initial_prompt,
    validate_export_request,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Extract prompt_rank/step_000000.npz from a completed run")
    export.add_argument("run")
    export.add_argument("--output", required=True)
    export.add_argument("--expect-seed", type=int, required=True)
    check = commands.add_parser("check", help="Validate a transferred initialization before loading a model")
    check.add_argument("path")
    check.add_argument("--tokens", type=int, required=True)
    destination = commands.add_parser("check-export", help="Preflight an initialization-only training run")
    destination.add_argument("output")
    destination.add_argument("--seed", type=int, required=True)
    destination.add_argument("--tokens", type=int, required=True)
    destination.add_argument("--reference")
    rename = commands.add_parser(
        "rename-libero-inits", help="Use explicit source names; keep legacy symlinks"
    )
    rename.add_argument("directory")
    rename.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    rename.add_argument("--tokens", type=int, default=2)
    resolve = commands.add_parser(
        "resolve-libero-init", help="Print an existing canonical/legacy P initializer path"
    )
    resolve.add_argument("directory")
    resolve.add_argument("--seed", type=int, required=True)
    resolve.add_argument("--tokens", type=int, default=2)
    args = parser.parse_args()
    if args.command == "export":
        metadata = export_initial_prompt(args.run, args.output, args.expect_seed)
    elif args.command == "check":
        _, metadata = read_initial_prompt(args.path, args.tokens)
    elif args.command == "rename-libero-inits":
        metadata = rename_libero_prompts(args.directory, args.seeds, args.tokens)
    elif args.command == "resolve-libero-init":
        print(resolve_libero_prompt(args.directory, args.seed, args.tokens))
        return
    else:
        reference = validate_export_request(args.output, args.reference, args.seed, args.tokens)
        metadata = {"export_preflight": "ok", "historical_reference_supplied": reference is not None}
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
