"""
Generate ULTRA-COMPACT CSV-style spatial training data.
Output format: `dir,label,dist_m,urgency,summary_word`
Example: `90,car,5,med,close`
Target: <15 tokens per response → 2s on Pi 5 Gemma 4 E2B.
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
LABELS = ["car", "truck", "bus", "person", "bike", "obstacle", "sound"]
URGENCIES = ["low", "med", "high", "crit"]
SUMMARIES = ["clear", "ahead", "close", "behind", "right", "left", "warn", "stop"]

SYSTEM_PROMPT = (
    "Pick CLOSEST object (lowest dist_m). Output 4 CSV: object_dir,label,dist_m,urgency. "
    "object_dir = the @X° from input (NOT heading). "
    "urgency: <2m=crit, <4m=high, <8m=med, else low. "
    "No JSON, no extra text, just CSV. "
    "Example: input 'heading=50 | visual=[car@90 5m, truck@180 12m]' → '90,car,5,med' "
    "(picked car because 5m < 12m, used 90° not 50°)"
)


def random_scenario() -> str:
    heading = random.randint(0, 359)
    n_obj = random.randint(1, 3)
    parts = []
    for _ in range(n_obj):
        label = random.choice(LABELS[:6])  # skip "sound" for input
        direction = random.randint(0, 359)
        dist = round(random.uniform(0.5, 25.0), 1)
        conf = round(random.uniform(0.3, 0.95), 2)
        parts.append(f"{label}@{direction}° {dist}m conf={conf}")
    return f"heading={heading}° | visual=[" + ", ".join(parts) + "]"


_CSV_RE = re.compile(
    r"^\s*(\d{1,3}),(car|truck|bus|person|bike|obstacle|sound),"
    r"(\d+(?:\.\d+)?),(low|med|high|crit)\s*$"
)


def query_teacher(prompt: str, retries: int = 2) -> str | None:
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 40,
        "temperature": 0.2,
        "stop": ["\n", "<end_of_turn>", "```"],
    }
    for _ in range(retries):
        try:
            r = requests.post(TEACHER_URL, json=body, timeout=30)
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  retry: {e}", file=sys.stderr)
            time.sleep(0.3)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", type=str, default="train.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    valid = invalid = 0
    t0 = time.monotonic()

    with out.open("w") as f:
        while valid < args.n:
            prompt = random_scenario()
            resp = query_teacher(prompt)
            if resp is None:
                continue
            # First line only
            first = resp.split("\n", 1)[0].strip()
            m = _CSV_RE.match(first)
            if not m:
                invalid += 1
                if invalid % 20 == 0:
                    print(f"  invalid: {invalid}, last: {first!r}", file=sys.stderr)
                continue
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": first},
                ]
            }
            f.write(json.dumps(example) + "\n")
            valid += 1
            if valid % 25 == 0:
                rate = valid / (time.monotonic() - t0)
                eta = (args.n - valid) / max(rate, 0.01)
                print(f"  {valid}/{args.n} valid, {invalid} invalid, {rate:.1f}/s, ETA {eta:.0f}s")

    print(f"Done. {valid} valid, {invalid} invalid in {time.monotonic()-t0:.0f}s")


if __name__ == "__main__":
    main()
