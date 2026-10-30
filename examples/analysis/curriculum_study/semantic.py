"""Optional frozen external PaliGemma MCQ probe on exported LIBERO states.

Human-provided questions, choices and answer labels are required. The VLM never
labels its own answers. This measures one EXTERNAL semantic proxy, not PI05's
true semantic capability. A full local VLM checkpoint is required, not tokenizer
files or a native PI05 checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .core import digest, read_json, write_json


def probabilities(log_scores):
    scores = np.asarray(log_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("Expected at least two finite candidate log likelihoods")
    scores -= scores.max()
    result = np.exp(scores)
    return result / result.sum()


def validate_questions(rows):
    seen, counts = set(), set()
    for row in rows:
        uid, choices, correct = row["state_uid"], row["choices"], row["correct_index"]
        if uid in seen or not uid:
            raise ValueError("Exactly one independently labelled question per state_uid is required")
        seen.add(uid)
        if (not isinstance(row["question"], str) or not row["question"].strip() or
                not isinstance(choices, list) or not 2 <= len(choices) <= 8 or
                any(not isinstance(x, str) or not x.strip() for x in choices) or
                len(set(choices)) != len(choices) or type(correct) is not int or not 0 <= correct < len(choices)):
            raise ValueError(f"Fill and verify question/choices/correct_index for {uid}")
        if row.get("rotate_degrees", 0) not in (0, 90, 180, 270):
            raise ValueError("Specify an explicit right-angle image rotation")
        counts.add(len(choices))
    if not rows or len(counts) != 1:
        raise ValueError("Use the same nonzero number of candidate choices for every state")


def make_template(states, output):
    rows = []
    for path in sorted(Path(states).glob("*/identity.json")):
        row = read_json(path)
        # No semantic labels are inferred from the task instruction.
        image = row["images"].get("image", next(iter(row["images"].values())))
        rows.append({"state_uid": row["state_uid"], "image": image, "rotate_degrees": 0,
                     "question": "", "choices": ["", "", "", ""], "correct_index": None})
    if not rows:
        raise ValueError("No exported state identities")
    with Path(output).open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def score(args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    validate_questions(questions)
    with args.features.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames, rows = reader.fieldnames, list(reader)
    mapping = {row["state_uid"]: row for row in rows}
    if len(mapping) != len(rows) or set(mapping) != {row["state_uid"] for row in questions}:
        raise ValueError("Questions and features must match state_uid one-to-one, with no missing states")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.model.is_dir():
        raise FileNotFoundError("Supply a full local external PaliGemma VLM checkpoint")
    cfg = read_json(args.model / "config.json")
    if cfg.get("model_type") != "paligemma":
        raise ValueError("This optional scorer supports Hugging Face PaliGemma checkpoints only")
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype).to(args.device).eval().requires_grad_(False)
    details = []
    with torch.inference_mode():
        for row in questions:
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = args.questions.parent / image_path
            with Image.open(image_path) as source:
                image = source.convert("RGB").rotate(row.get("rotate_degrees", 0), expand=True)
            labels = [chr(ord("A") + i) for i in range(len(row["choices"]))]
            question = row["question"].strip() + "\n" + "\n".join(
                f"{label}. {option}" for label, option in zip(labels, row["choices"], strict=True))
            question += "\nAnswer with the option letter only.\n"
            log_scores, lengths = [], []
            for label in labels:
                inputs = processor(images=image, text=question, suffix=label, return_tensors="pt").to(args.device)
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
                # Same suffix + EOS scoring for every option; never score the prompt.
                n = int((inputs["labels"][:, 1:] != -100).sum().item())
                if n < 1:
                    raise ValueError("Processor produced no answer labels")
                outputs = model(**inputs)
                loss = float(outputs.loss.float().item())
                if not math.isfinite(loss):
                    raise FloatingPointError("Nonfinite semantic probe likelihood")
                log_scores.append(-loss * n)
                lengths.append(n)
                del outputs, inputs
            p = probabilities(log_scores)
            difficulty = -math.log(max(float(p[row["correct_index"]]), np.finfo(float).tiny))
            mapping[row["state_uid"]]["semantic_difficulty"] = difficulty
            details.append({"state_uid": row["state_uid"], "question": question, "correct_index": row["correct_index"],
                            "candidate_log_likelihoods": log_scores, "candidate_label_tokens": lengths,
                            "candidate_probabilities": p.tolist(), "semantic_difficulty": difficulty,
                            "image_sha256": digest(image_path)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output.with_suffix(".semantic.json"), {
        "model": str(args.model.resolve()), "questions_sha256": digest(args.questions),
        "source_features_sha256": digest(args.features), "details": details,
        "definition": "negative log probability of labelled correct option, normalized over fixed candidates",
        "scope": "External frozen VLM semantic probe; not a calibrated success probability or PI05 semantics"}, exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template")
    template.add_argument("--states", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("score")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--questions", type=Path, required=True)
    run.add_argument("--features", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "template":
        make_template(args.states, args.output)
    else:
        score(args)
    print(args.output)


if __name__ == "__main__":
    main()
