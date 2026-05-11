"""
Generate CJK-encoded spatial training data with SMOOTH angle precision.
Format per object: [T][O][L][U]  (4 chars)
  T = angle tens (0-18, 19 chars)
  O = angle ones (0-18, 19 chars)
  Angle = T*19 + O, clamped 0-359
  L = label (car/truck/bus/person/bike/obstacle)
  U = urgency (crit/high/med/low)
2 objects = 8 chars total.
"""
import argparse, json, random, re, time

T_CHARS = list("一二三四五六七八九十甲乙丙丁戊己庚辛壬")  # 19
O_CHARS = list("子丑寅卯辰巳午未申酉戌亥角亢氐房心尾箕")  # 19
LABELS_CHAR = {"car":"车","truck":"卡","bus":"巴","person":"人","bike":"自","obstacle":"物"}
URG_CHAR = {"crit":"危","high":"急","med":"中","low":"远"}
ALL_LABELS = list(LABELS_CHAR.keys())

assert len(T_CHARS) == 19 and len(O_CHARS) == 19


def deg_to_chars(deg: float) -> str:
    d = int(round(deg)) % 360
    t = d // 19
    o = d % 19
    if t > 18: t = 18
    return T_CHARS[t] + O_CHARS[o]


def chars_to_deg(s: str) -> int:
    if len(s) < 2: return 0
    try:
        t = T_CHARS.index(s[0])
        o = O_CHARS.index(s[1])
    except ValueError:
        return 0
    return min(359, t * 19 + o)


def dist_to_urg(dist: float) -> str:
    if dist < 2: return "危"
    if dist < 4: return "急"
    if dist < 8: return "中"
    return "远"


def random_scene():
    heading = random.randint(0, 359)
    n = random.randint(2, 5)
    objs = []
    for _ in range(n):
        label = random.choice(ALL_LABELS)
        # Sample T and O uniformly to guarantee even O-char distribution
        t = random.randint(0, 18)
        o = random.randint(0, 18)
        d = min(t * 19 + o, 359)
        m = round(random.uniform(0.5, 25.0), 1)
        objs.append((label, d, m))
    objs.sort(key=lambda x: x[2])
    primary = objs[0]
    secondary = objs[1] if len(objs) > 1 else objs[0]
    prompt_parts = [f"{lbl}@{d}° {m}m" for (lbl, d, m) in objs]
    user = f"heading={heading}° | visual=[" + ", ".join(prompt_parts) + "]"
    answer = (
        deg_to_chars(primary[1])  + LABELS_CHAR[primary[0]]  + dist_to_urg(primary[2])
      + deg_to_chars(secondary[1]) + LABELS_CHAR[secondary[0]] + dist_to_urg(secondary[2])
    )
    return user, answer


SYSTEM_PROMPT = (
    "Output 8 CJK chars encoding 2 closest objects. Format: T1O1L1U1 T2O2L2U2 (no spaces). "
    "Angle: T(tens) O(ones), deg=T*19+O. "
    "T alphabet: 一二三四五六七八九十甲乙丙丁戊己庚辛壬 (idx 0-18). "
    "O alphabet: 子丑寅卯辰巳午未申酉戌亥角亢氐房心尾箕 (idx 0-18). "
    "L: 车卡巴人自物 (car truck bus person bike obstacle). "
    "U: 危急中远 (crit<2m, high<4m, med<8m, low>=8m). "
    "Pick 2 closest objects. No other text."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", type=str, default="train_cjk_smooth.jsonl")
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
            if (i+1) % 500 == 0:
                print(f"  {i+1}/{args.n}")
    print(f"Done {args.n} in {time.monotonic()-t0:.1f}s")


if __name__ == "__main__":
    main()
