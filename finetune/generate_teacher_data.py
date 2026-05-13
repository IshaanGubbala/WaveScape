"""
Generate teacher distillation dataset for Gemma 3 1B fine-tuning.
Queries Gemma 4 E2B (running on Mac via llama-server :8082) with synthetic
spatial scenarios. Saves valid (input, JSON output) pairs as JSONL.

Run: python3 generate_teacher_data.py --n 500 --out train.jsonl
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

TEACHER_URL = "http://localhost:8082/v1/chat/completions"

LABELS = ["car", "truck", "bus", "person", "bike", "sound_source", "obstacle"]
SOUNDS = ["engine", "horn", "footsteps", "speech", "none"]

SYSTEM_PROMPT = (
    "Spatial JSON. Max 1 object. 2-4 word summary. "
    "dir 0=ahead 90=R 180=back 270=L. <3m=high <6m=med far=low. "
    'Output: {"objects":[{"dir":N,"label":"car","dist_m":N}],'
    '"dominant_hazard":{"dir":N,"urgency":"med"},"summary":"short"}'
)


def random_scenario() -> str:
    """Build a random `heading=X | visual=[...]` input string."""
    heading = random.randint(0, 359)
    n_obj = random.randint(1, 4)
    parts = []
    for _ in range(n_obj):
        label = random.choice(LABELS)
        direction = random.randint(0, 359)
        dist = round(random.uniform(0.5, 25.0), 1)
        conf = round(random.uniform(0.3, 0.95), 2)
        sound = random.choice(SOUNDS)
        snd_str = f" snd={sound}" if sound != "none" and random.random() < 0.4 else ""
        parts.append(f"{label}@{direction}° {dist}m conf={conf}{snd_str}")
    visual = "[" + ", ".join(parts) + "]" if parts else "[]"
    return f"heading={heading}° | visual={visual}"


def query_teacher(prompt: str, retries: int = 2) -> str | None:
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 120,
        "temperature": 0.2,
        "stop": ["\n\n", "<end_of_turn>", "```"],
    }
    for _ in range(retries):
        try:
            r = requests.post(TEACHER_URL, json=body, timeout=60)
            content = r.json()["choices"][0]["message"]["content"].strip()
            return content
        except Exception as e:
            print(f"  retry: {e}", file=sys.stderr)
            time.sleep(0.5)
    return None


def extract_json(text: str) -> dict | None:
    """Extract first valid JSON object from teacher response."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text, m.start())
            if isinstance(data, dict) and "objects" in data:
                return data
        except json.JSONDecodeError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="Number of examples to generate")
    ap.add_argument("--out", type=str, default="train.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)

    valid = 0
    invalid = 0
    skipped = 0
    t0 = time.monotonic()

    with out_path.open("w") as f:
        while valid < args.n:
            prompt = random_scenario()
            response = query_teacher(prompt)
            if response is None:
                skipped += 1
                continue
            parsed = extract_json(response)
            if parsed is None:
                invalid += 1
                if invalid % 10 == 0:
                    print(f"  invalid responses: {invalid}", file=sys.stderr)
                continue
            # Re-serialize to enforce compact format
            clean_json = json.dumps(parsed, separators=(",", ":"))
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": clean_json},
                ]
            }
            f.write(json.dumps(example) + "\n")
            valid += 1
            if valid % 25 == 0:
                rate = valid / (time.monotonic() - t0)
                print(f"  {valid}/{args.n} valid, {invalid} invalid, "
                      f"{rate:.1f}/s, ETA {(args.n - valid) / max(rate, 0.01):.0f}s")

    elapsed = time.monotonic() - t0
    print(f"\nDone. {valid} valid, {invalid} invalid, {skipped} skipped in {elapsed:.0f}s")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
