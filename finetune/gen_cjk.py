"""
Generate CJK-encoded spatial training data.
Output: 6 chars = 6 tokens = ~1s on Pi 5.
Format: <dir1><label1><urg1><dir2><label2><urg2>
"""
import argparse, json, random, re, sys, time
from pathlib import Path
import requests

TEACHER_URL = "http://localhost:8082/v1/chat/completions"

# Codebook
DIRS_CHAR = ["北","艮","东","巽","南","坤","西","乾"]  # N,NE,E,SE,S,SW,W,NW
LABELS_CHAR = {"car":"车","truck":"卡","bus":"巴","person":"人","bike":"自","obstacle":"物"}
URG_CHAR = {"crit":"危","high":"急","med":"中","low":"远"}
ALL_LABELS = list(LABELS_CHAR.keys())


def deg_to_dir_char(deg: float) -> str:
    # 8 buckets, 45° each, 0=N centered. N covers 337.5-22.5
    bucket = int(((deg + 22.5) % 360) // 45)
    return DIRS_CHAR[bucket]


def dist_to_urg(dist: float) -> str:
    if dist < 2: return "危"
    if dist < 4: return "急"
    if dist < 8: return "中"
    return "远"


def random_scene():
    heading = random.randint(0, 359)
    n = random.randint(2, 4)
    objs = []
    for _ in range(n):
        label = random.choice(ALL_LABELS)
        d = random.randint(0, 359)
        m = round(random.uniform(0.5, 25.0), 1)
        objs.append((label, d, m))
    # Sort by distance (closest first), take 2
    objs.sort(key=lambda x: x[2])
    primary = objs[0]
    secondary = objs[1] if len(objs) > 1 else objs[0]
    prompt_parts = [f"{lbl}@{d}° {m}m" for (lbl, d, m) in objs]
    user = f"heading={heading}° | visual=[" + ", ".join(prompt_parts) + "]"
    answer = (
        deg_to_dir_char(primary[1]) + LABELS_CHAR[primary[0]] + dist_to_urg(primary[2])
        + deg_to_dir_char(secondary[1]) + LABELS_CHAR[secondary[0]] + dist_to_urg(secondary[2])
    )
    return user, answer


SYSTEM_PROMPT = (
    "Output 6 CJK chars encoding 2 closest objects. Format: D1L1U1D2L2U2. "
    "D=direction 北艮东巽南坤西乾 (N NE E SE S SW W NW). "
    "L=label 车卡巴人自物 (car truck bus person bike obstacle). "
    "U=urgency 危急中远 (crit high med low, by distance). "
    "Pick 2 closest objects (lowest dist_m). No other text."
)

_RE = re.compile(r"^[北艮东巽南坤西乾][车卡巴人自物][危急中远][北艮东巽南坤西乾][车卡巴人自物][危急中远]$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", type=str, default="train_cjk.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    t0 = time.monotonic()
    with open(args.out, "w") as f:
        for i in range(args.n):
            user, gold = random_scene()
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": gold},
                ]
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            if (i+1) % 100 == 0:
                print(f"  {i+1}/{args.n}")
    print(f"Done {args.n} in {time.monotonic()-t0:.1f}s")


if __name__ == "__main__":
    main()
